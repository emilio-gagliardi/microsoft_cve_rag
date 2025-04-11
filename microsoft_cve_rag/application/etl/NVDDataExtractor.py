"""NVD Data Extractor for retrieving vulnerability information from the National Vulnerability Database.

This module provides functionality to extract detailed vulnerability information from
the NVD website using Selenium WebDriver. It handles the extraction of CVSS scores,
metrics, CWE data, and other vulnerability-related information.
"""

import json
import logging
import math
# import os
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime
from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

# from microsoft_cve_rag.application.etl.type_utils import convert_to_float

logging.getLogger(__name__)


@dataclass
class ScrapingParams:
    """Parameters for controlling web scraping behavior.

    Attributes:
        batch_size (int): Number of records to process in each batch
        delay_between_batches (float): Delay in seconds between batches
        hover_wait (float): Wait time in seconds after hover action
        max_retries (int): Maximum number of retry attempts
        retry_delay (float): Delay in seconds between retries
    """

    batch_size: int = 100
    delay_between_batches: float = 5.0
    hover_wait: float = 1.0
    max_retries: int = 3
    retry_delay: float = 2.0

    @classmethod
    def from_target_time(
        cls, num_cves: int, target_time_per_cve: float = 10.0
    ) -> 'ScrapingParams':
        """Calculate optimal scraping parameters based on desired processing time.

        Uses a logarithmic scaling to increase delays as batch size grows.

        Args:
            num_cves (int): Number of CVEs to process.
            target_time_per_cve (float): Target processing time per CVE in seconds.

        Returns:
            ScrapingParams: Configured scraping parameters.
        """
        base_batch_size = 100
        base_delay_between_batches = 5.0
        base_hover_wait = 1.0
        base_max_retries = 3
        base_retry_delay = 2.0

        scale = math.log2(max(num_cves, 2)) / 4

        return cls(
            batch_size=min(base_batch_size * scale, 200),
            delay_between_batches=min(base_delay_between_batches * scale, 10.0),
            hover_wait=min(base_hover_wait * scale, 2.0),
            max_retries=min(base_max_retries * scale, 5),
            retry_delay=min(base_retry_delay * scale, 4.0),
        )

    def estimate_time_per_cve(self) -> float:
        """Estimate the processing time for a single CVE.

        Returns:
            float: Estimated time in seconds to process one CVE.
        """
        base_time = (
            self.delay_between_batches
            + self.hover_wait
            + self.retry_delay
        )
        selenium_overhead = 1.0
        network_latency = 2.0
        parsing_time = 1.0
        tooltip_wait = 1.0

        return (
            base_time
            + selenium_overhead
            + network_latency
            + parsing_time
            + tooltip_wait
        )

    def estimate_total_time(self, num_cves: int) -> float:
        """Estimate total processing time for a batch of CVEs.

        Args:
            num_cves (int): Number of CVEs to process.

        Returns:
            float: Estimated total processing time in seconds.
        """
        return self.estimate_time_per_cve() * num_cves


class NVDDataExtractor:
    """Extracts vulnerability data from the National Vulnerability Database website.

    This class handles the extraction of various vulnerability metrics, scores, and
    related information from NVD using Selenium WebDriver. It supports extraction of
    CVSS scores, vector metrics, CWE data, and other vulnerability attributes.

    Attributes:
        ELEMENT_MAPPINGS (dict): XPath mappings for basic page elements.
        VECTOR_SOURCES (dict): XPath mappings for different vector sources.
        METRIC_PATTERNS (dict): XPath patterns for extracting metric values.
    """

    # Element mappings as class constant
    ELEMENT_MAPPINGS = {
        'nvd_published_date': {
            'xpath': "//span[@data-testid='vuln-published-on']",
        },
        'nvd_description': {
            'xpath': "//p[@data-testid='vuln-description']",
        },
        'base_score': {
            'xpath': "//span[@data-testid='vuln-cvssv3-base-score']",
        },
        'base_score_num': {
            'xpath': "//span[@data-testid='vuln-cvssv3-base-score']",
        },
        'base_score_rating': {
            'xpath': "//span[@data-testid='vuln-cvssv3-base-score-severity']",
        },
        'vector_element': {
            'xpath': "//span[@data-testid='vuln-cvssv3-vector']",
        },
        'vector': {
            'xpath': "//span[@data-testid='vuln-cvssv3-vector']",
        },
        'impact_score': {
            'xpath': "//span[@data-testid='vuln-cvssv3-impact-score']",
        },
        'exploitability_score': {
            'xpath': "//span[@data-testid='vuln-cvssv3-exploitability-score']",
        },
        'attack_vector': {
            'xpath': "//span[@data-testid='vuln-cvssv3-av']",
        },
        'attack_complexity': {
            'xpath': "//span[@data-testid='vuln-cvssv3-ac']",
        },
        'privileges_required': {
            'xpath': "//span[@data-testid='vuln-cvssv3-pr']",
        },
        'user_interaction': {
            'xpath': "//span[@data-testid='vuln-cvssv3-ui']",
        },
        'scope': {
            'xpath': "//span[@data-testid='vuln-cvssv3-s']",
        },
        'confidentiality': {
            'xpath': "//span[@data-testid='vuln-cvssv3-c']",
        },
        'integrity': {
            'xpath': "//span[@data-testid='vuln-cvssv3-i']",
        },
        'availability': {
            'xpath': "//span[@data-testid='vuln-cvssv3-a']",
        },
        # CWE mappings
        'cwe_id': {
            'xpath': "//td[contains(@data-testid, 'vuln-CWEs-link-')]",
        },
        'cwe_name': {
            'xpath': "//td[contains(@data-testid, 'vuln-CWEs-link-')]/following-sibling::td[1]",
        },
        'cwe_source': {
            'xpath': "//td[contains(@data-testid, 'vuln-CWEs-link-')]/following-sibling::td[2]",
        },
        'cwe_url': {
            'xpath': "//td[contains(@data-testid, 'vuln-CWEs-link-')]/a",
            'attribute': 'href'
        }
    }
    VECTOR_SOURCES = {
        'nist': {
            'prefix': 'nist_',
            'vector_element': "//span[@data-testid='vuln-cvss3-nist-vector']",
            'base_score': "//span[@data-testid='vuln-cvss3-nist-panel-score']",
        },
        'cna': {
            'prefix': 'cna_',
            'vector_element': "//span[@data-testid='vuln-cvss3-cna-vector']",
            'base_score': "//span[@data-testid='vuln-cvss3-cna-panel-score']",
        },
        'adp': {
            'prefix': 'adp_',
            'vector_element': "//span[@data-testid='vuln-cvss3-adp-vector']",
            'base_score': "//span[@data-testid='vuln-cvss3-adp-panel-score']",
        },
    }
    METRIC_PATTERNS = {
        'base_score': (
            ".//p/strong[contains(text(), 'Base"
            " Score:')]/following-sibling::span[1]"
        ),
        'base_score_rating': (
            ".//p/strong[contains(text(), 'Base"
            " Score:')]/following-sibling::span[2]"
        ),
        'vector': (
            ".//p/strong[contains(text(), 'Vector:')]/following-sibling::span"
        ),
        'impact_score': (
            ".//p/strong[contains(text(), 'Impact"
            " Score:')]/following-sibling::span"
        ),
        'exploitability_score': (
            ".//p/strong[contains(text(), 'Exploitability"
            " Score:')]/following-sibling::span"
        ),
        'attack_vector': (
            ".//p/strong[contains(text(), 'Attack"
            " Vector')]/following-sibling::span"
        ),
        'attack_complexity': (
            ".//p/strong[contains(text(), 'Attack"
            " Complexity')]/following-sibling::span"
        ),
        'privileges_required': (
            ".//p/strong[contains(text(), 'Privileges"
            " Required')]/following-sibling::span"
        ),
        'user_interaction': (
            ".//p/strong[contains(text(), 'User"
            " Interaction')]/following-sibling::span"
        ),
        'scope': (
            ".//p/strong[contains(text(), 'Scope')]/following-sibling::span"
        ),
        'confidentiality': (
            ".//p/strong[contains(text(),"
            " 'Confidentiality')]/following-sibling::span"
        ),
        'integrity': (
            ".//p/strong[contains(text(),"
            " 'Integrity')]/following-sibling::span"
        ),
        'availability': (
            ".//p/strong[contains(text(),"
            " 'Availability')]/following-sibling::span"
        ),
    }

    HIDDEN_INPUT_IDS = {
        'cna': 'cnaV3MetricHidden',
        'nist': 'nistV3MetricHidden',
        'adp': 'adpV3MetricHidden'
    }

    HIDDEN_INPUT_METRIC_TEST_IDS = {
        'base_score_num': 'vuln-cvssv3-base-score',
        'base_score_rating': 'vuln-cvssv3-base-score-severity',
        'vector': 'vuln-cvssv3-vector',
        'impact_score': 'vuln-cvssv3-impact-score',
        'exploitability_score': 'vuln-cvssv3-exploitability-score',
        # --- Additions for detailed vector components ---
        'attack_vector': 'vuln-cvssv3-av',
        'attack_complexity': 'vuln-cvssv3-ac',
        'privileges_required': 'vuln-cvssv3-pr',
        'user_interaction': 'vuln-cvssv3-ui',
        'scope': 'vuln-cvssv3-s',
        'confidentiality': 'vuln-cvssv3-c',
        'integrity': 'vuln-cvssv3-i',
        'availability': 'vuln-cvssv3-a',
    }

    def __init__(
        self,
        properties_to_extract: Optional[List[str]] = None,
        max_records: Optional[int] = None,
        headless: bool = True,
        window_size: Optional[Tuple[int, int]] = (1240, 1080),
        scraping_params: Optional[ScrapingParams] = None,
        show_progress: bool = False,
    ) -> None:
        """Initialize the NVD Data Extractor.

        Args:
            properties_to_extract (Optional[List[str]]): List of properties to
                extract. If None, extracts all valid properties.
            max_records (Optional[int]): Maximum number of records to process.
            headless (bool): Whether to run Chrome in headless mode.
            window_size (Optional[Tuple[int, int]]): Browser window dimensions.
            scraping_params (Optional[ScrapingParams]): Custom scraping parameters.
            show_progress (bool): Whether to display a progress bar.
        """
        # --- CORRECTED DEFINITION of valid_properties ---
        # Use the static method to get the *actual* list of all possible output columns
        self.valid_properties = set(NVDDataExtractor.get_all_possible_columns())
        # --- END CORRECTION ---

        # Validate and normalize properties to extract (This logic can now stay)
        if properties_to_extract:
            # Check against the now correctly defined self.valid_properties
            invalid_props = set(properties_to_extract) - self.valid_properties
            if invalid_props:
                # This warning should no longer appear if the passed list is correct
                logging.warning(
                    f"Ignoring invalid properties: {invalid_props}"
                )
            # Keep only the properties that are valid according to get_all_possible_columns
            self.properties_to_extract = [
                p for p in properties_to_extract if p in self.valid_properties
            ]
            # Ensure properties_to_extract isn't empty if filtering happened
            if not self.properties_to_extract and properties_to_extract:
                logging.error("All requested properties were deemed invalid. Check property names.")
                # Decide how to handle this - raise error or default to all?
                # Defaulting to all might be safer if the input list was just wrong.
                self.properties_to_extract = list(self.valid_properties)
            elif not self.properties_to_extract and not properties_to_extract:
                # Case where properties_to_extract was None or empty initially
                self.properties_to_extract = list(self.valid_properties)

        else:
            # If no specific list is provided, default to extracting all valid properties
            self.properties_to_extract = list(self.valid_properties)

        # Log the final list of properties that will be extracted
        logging.debug(f"NVD Extractor will target these properties: {self.properties_to_extract}")

        self.max_records = max_records
        # Use default ScrapingParams if none provided
        self.params = scraping_params or ScrapingParams()
        self.headless = headless
        self.window_size = window_size
        self.show_progress = show_progress
        self.driver = None
        self._last_request_time = 0
        # Call setup_driver (ensure it handles potential errors)
        try:
            self.setup_driver()
        except Exception as e:
            logging.error(f"Failed to setup WebDriver during initialization: {e}", exc_info=True)
            # Depending on requirements, you might want to raise e here
            # or allow the instance to exist without a driver (though it would be unusable).
            self.driver = None

    def setup_driver(self) -> None:
        """Setup the Selenium WebDriver instance."""
        chrome_options = webdriver.ChromeOptions()
        if self.headless:
            chrome_options.add_argument('--headless=new')
        chrome_options.add_argument(
            f'--window-size={self.window_size[0]},{self.window_size[1]}'
        )
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.page_load_strategy = 'eager'
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_argument('--silent')
        chrome_options.add_experimental_option(
            'excludeSwitches', ['enable-logging', 'enable-automation']
        )

        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(self.params.hover_wait)

    def cleanup(self) -> None:
        """Cleanup the Selenium WebDriver instance."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logging.debug("Selenium WebDriver cleaned up.")
            except Exception as e:
                logging.error(f"Error during driver cleanup: {str(e)}")

    def _switch_to_cvss3(self) -> None:
        """Switch to CVSS 3.x tab if it exists and is not already active."""
        try:
            cvss3_button = self.driver.find_element(
                By.XPATH,
                "//button[@id='btn-cvss3' and contains(@title, 'CVSS 3.x')]"
            )
            if 'btn-active' not in cvss3_button.get_attribute('class'):
                cvss3_button.click()
                time.sleep(0.5)  # Wait for tab switch
                logging.debug("Switched to CVSS 3.x tab")
        except NoSuchElementException:
            logging.debug("CVSS 3.x tab not found or already active")
        except Exception as e:
            logging.error(f"Error switching to CVSS 3.x tab: {str(e)}")

    def _extract_base_score(
        self, score_text: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Extract numeric score and rating from score text.

        Args:
            score_text (str): Text containing score and rating (e.g. '7.5 HIGH').

        Returns:
            Tuple[Optional[float], Optional[str]]: Tuple of (score, rating).
        """
        try:
            if not score_text or score_text.lower() == 'none':
                return None, None

            parts = score_text.strip().split(' ', 1)
            score = float(parts[0]) if parts[0] else None
            rating = parts[1].lower() if len(parts) > 1 else 'none'
            return score, rating
        except (ValueError, IndexError) as e:
            logging.warning(
                f"Error parsing base score '{score_text}': {str(e)}"
            )
            return None, 'none'

    def _set_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set appropriate data types for DataFrame columns.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with correct column types.
        """
        type_mappings = {
            # Numeric columns
            'base_score_num': 'float64',
            'impact_score': 'float64',
            'exploitability_score': 'float64',
            # Date columns handled separately
            'nvd_published_date': 'datetime',
            # String columns (categorical might be more appropriate for some)
            'base_score_rating': 'category',
            'attack_vector': 'category',
            'attack_complexity': 'category',
            'privileges_required': 'category',
            'user_interaction': 'category',
            'scope': 'category',
            'confidentiality': 'category',
            'integrity': 'category',
            'availability': 'category',
            'nvd_description': 'string',
            'cwe_id': 'string',
            'cwe_name': 'string',
            'cwe_source': 'category',
            'cwe_url': 'string',
        }

        def parse_nvd_date(date_str):
            if pd.isna(date_str):
                return None
            try:
                # First try MM/DD/YYYY format (from NVD website)
                return datetime.strptime(str(date_str), '%m/%d/%Y')
            except ValueError:
                try:
                    # Try YYYY-MM-DD HH:MM:SS format (from MongoDB)
                    return datetime.strptime(
                        str(date_str), '%Y-%m-%d %H:%M:%S'
                    )
                except ValueError as e:
                    logging.warning(
                        f"Could not parse NVD date: {date_str} - {str(e)}"
                    )
                    return None

        # Apply source prefixes to relevant columns
        prefixed_types = {}
        for source in self.VECTOR_SOURCES.keys():
            prefix = self.VECTOR_SOURCES[source]['prefix']
            for col, dtype in type_mappings.items():
                if col in ['base_score_num', 'base_score_rating'] + list(
                    self.METRIC_PATTERNS.keys()
                ):
                    prefixed_types[f"{prefix}{col}"] = dtype

        # Merge original and prefixed type mappings
        type_mappings.update(prefixed_types)

        # Convert types and handle errors
        for column, dtype in type_mappings.items():
            if column in df.columns:
                try:
                    if dtype == 'datetime':
                        # Convert to Python datetime objects
                        df[column] = df[column].apply(parse_nvd_date)
                    elif dtype == 'category':
                        df[column] = df[column].astype(str).astype('category')
                    else:
                        df[column] = df[column].astype(dtype)
                except Exception as e:
                    logging.warning(
                        f"Failed to convert column {column} to {dtype}:"
                        f" {str(e)}"
                    )

        return df

    def _extract_from_hidden_input(self, source: str) -> Dict[str, Any]:
        """Extract metrics from hidden input field for a given source,
           including detailed vector components and constructing base_score.
           Includes detailed logging for base score components.

        Args:
            source: The source type ('cna', 'nist', or 'adp').

        Returns:
            Dict containing the extracted metrics with appropriate prefixes.
        """
        prefix = self.VECTOR_SOURCES[source]['prefix']
        data = {f"{prefix}{key}": None for key in self.HIDDEN_INPUT_METRIC_TEST_IDS.keys()}
        data[f"{prefix}base_score"] = None

        try:
            input_id = self.HIDDEN_INPUT_IDS.get(source)
            if not input_id:
                logging.warning(f"No hidden input ID configured for source: {source}")
                return data

            hidden_input = self.driver.find_element(By.ID, input_id)
            raw_html = hidden_input.get_attribute("value")

            if not raw_html:
                logging.debug(f"Hidden input for {source} (ID: {input_id}) found but is empty.")
                return data

            soup = BeautifulSoup(raw_html, "html.parser")
            logging.debug(f"Processing hidden input for {source}...")

            score_num_val = None
            score_rating_val = None

            for key, test_id in self.HIDDEN_INPUT_METRIC_TEST_IDS.items():
                element = soup.find("span", {"data-testid": test_id})
                value = None
                value_text = "ELEMENT_NOT_FOUND"

                # --- TARGETED LOGGING ---
                is_base_score_component = key in ['base_score_num', 'base_score_rating']
                if is_base_score_component:
                    logging.debug(f"[{source}-{key}] Searching for test_id: '{test_id}'...")
                # --- END TARGETED LOGGING ---

                if element:
                    value_text = element.text.strip()
                    prefixed_key = f"{prefix}{key}"

                    # --- TARGETED LOGGING ---
                    if is_base_score_component:
                        logging.debug(f"[{source}-{key}] Found element! Raw text: '{value_text}'")
                    # --- END TARGETED LOGGING ---

                    if key in ['base_score_num', 'impact_score', 'exploitability_score']:
                        value = self._convert_to_float(value_text)
                        if key == 'base_score_num':
                            score_num_val = value
                            # --- TARGETED LOGGING ---
                            logging.debug(f"[{source}-{key}] Processed value (float): {value}. Stored in score_num_val.")
                            # --- END TARGETED LOGGING ---
                    elif key == 'base_score_rating':
                        value = value_text.lower() if value_text else None
                        score_rating_val = value_text
                        # --- TARGETED LOGGING ---
                        logging.debug(f"[{source}-{key}] Processed value (lower): {value}. Stored original '{value_text}' in score_rating_val.")
                        # --- END TARGETED LOGGING ---
                    else:
                        # Process other keys as before
                        value = value_text if value_text else None

                    data[prefixed_key] = value
                    # --- TARGETED LOGGING ---
                    # if is_base_score_component:
                    #      logging.info(f"[{source}-{key}] Assigned value '{value}' to data['{prefixed_key}']")
                    # --- END TARGETED LOGGING ---

                # --- TARGETED LOGGING ---
                elif is_base_score_component:  # Log if element was *not* found for base score keys
                    logging.warning(f"[{source}-{key}] Element with test_id '{test_id}' NOT FOUND in hidden input HTML.")
                # --- END TARGETED LOGGING ---

            # Construct the combined base_score string (logic remains the same)
            if score_num_val is not None and score_rating_val is not None:
                data[f"{prefix}base_score"] = f"{score_num_val} {score_rating_val.upper()}"
                logging.debug(f"[{source}] Constructed combined base_score: '{data[f'{prefix}base_score']}'")
            else:
                logging.warning(f"[{source}] Could not construct combined base_score. Num: {score_num_val}, Rating: {score_rating_val}")

        except NoSuchElementException:
            logging.debug(f"Hidden input field with ID '{self.HIDDEN_INPUT_IDS.get(source)}' not found for source '{source}'.")
        except Exception as e:
            logging.error(f"Error parsing hidden input for {source} (ID: {self.HIDDEN_INPUT_IDS.get(source)}): {str(e)}", exc_info=True)

        # Log the state of the specific keys before returning
        logging.debug(f"[{source}] Returning data. Keys check: "
                     f"{prefix}base_score_num='{data.get(f'{prefix}base_score_num')}', "
                     f"{prefix}base_score_rating='{data.get(f'{prefix}base_score_rating')}', "
                     f"{prefix}base_score='{data.get(f'{prefix}base_score')}'")
        return data

    def _convert_to_float(self, value: str) -> Optional[float]:
        """Convert string to float, handling None and empty strings.

        Args:
            value: String value to convert

        Returns:
            Float value or None if conversion fails
        """
        try:
            return float(value.strip()) if value and value.strip() else None
        except (ValueError, AttributeError, TypeError):
            return None

    def extract_vector_metrics(self) -> Dict[str, Any]:
        """Extract vector metrics using hidden input first, then hover method.

        Returns:
            Dictionary containing all extracted metrics
        """
        # Switch to CVSS 3.x tab if needed
        self._switch_to_cvss3()

        data = {}

        # Get vector string from hidden input
        vector_element = self.extract_property('vector')
        if vector_element:
            # Parse vector string
            vector_parts = vector_element.split('/')
            for part in vector_parts:
                if ':' in part:
                    metric, value = part.split(':')
                    if metric in self.ELEMENT_MAPPINGS:
                        data[metric] = value

        # Get base score from hidden input
        base_score = self.extract_property('base_score')
        if base_score:
            data['base_score'] = self._convert_to_float(base_score)

        # Get impact and exploitability scores
        impact_score = self.extract_property('impact_score')
        if impact_score:
            data['impact_score'] = self._convert_to_float(impact_score)

        exploitability_score = self.extract_property('exploitability_score')
        if exploitability_score:
            data['exploitability_score'] = self._convert_to_float(exploitability_score)

        return data

    def extract_property(self, property_name: str) -> Optional[str]:
        """Extract a property from the page using predefined element mappings.

        Args:
            property_name: Name of the property to extract

        Returns:
            Optional[str]: Extracted property value or None if not found
        """
        try:
            if property_name not in self.ELEMENT_MAPPINGS:
                return None

            mapping = self.ELEMENT_MAPPINGS[property_name]
            element = self.driver.find_element(
                By.XPATH, mapping['xpath']
            )

            if 'attribute' in mapping:
                value = element.get_attribute(mapping['attribute'])
            else:
                value = element.text

            if property_name == 'nvd_published_date':
                try:
                    # Parse MM/DD/YYYY format
                    parsed_date = datetime.strptime(value.strip(), '%m/%d/%Y')
                    # Convert to YYYY-MM-DD HH:MM:SS format
                    return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError as e:
                    logging.warning(
                        f"Could not parse NVD date: {value} - {str(e)}"
                    )
                    return None

            return value.strip() if value else None

        except NoSuchElementException:
            logging.debug(f"Element not found for property: {property_name}")
            return None
        except Exception as e:
            logging.error(f"Error extracting {property_name}: {str(e)}")
            return None

    def extract_cwe_data(self) -> List[Dict[str, Any]]:
        """Extract CWE (Common Weakness Enumeration) data from the page.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing CWE information.
        """
        cwe_data = []
        try:
            # Find the CWE table
            table = self.driver.find_element(
                By.XPATH, "//table[@data-testid='vuln-CWEs-table']"
            )

            # Find all rows except header
            rows = table.find_elements(By.XPATH, ".//tbody/tr")

            for row in rows:
                try:
                    # Get all cells in the row
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 3:
                        continue

                    id_cell = cells[0]
                    name_cell = cells[1]
                    source_cell = cells[2]

                    cwe_name = name_cell.text.strip()

                    # Skip "Insufficient Information" entries
                    if cwe_name.lower() == "insufficient information":
                        continue

                    cwe_entry = {
                        'cwe_id': None,
                        'cwe_url': None,
                        'cwe_name': cwe_name,
                        'cwe_source': source_cell.text.strip(),
                    }

                    # Look for anchor tag in ID cell
                    anchor = id_cell.find_elements(By.TAG_NAME, "a")
                    if anchor:
                        cwe_entry['cwe_id'] = anchor[0].text.strip()
                        cwe_entry['cwe_url'] = anchor[0].get_attribute('href')
                    else:
                        cwe_entry['cwe_id'] = id_cell.text.strip()

                    cwe_data.append(cwe_entry)

                except Exception as e:
                    logging.warning(f"Error processing CWE row: {str(e)}")
                    continue

        except NoSuchElementException:
            logging.debug("No CWE table found")
        except Exception as e:
            logging.error(f"Error extracting CWE data: {str(e)}")

        return cwe_data

    def extract_data_from_url(self, url: str) -> Dict[str, Any]:
        """Extract data from the given URL using hidden inputs for metrics
           and direct properties for others.

        Args:
            url (str): URL to extract data from.

        Returns:
            Dict[str, Any]: Dictionary containing extracted data.
        """
        # Initialize data dictionary - crucial for consistent structure
        # You might want to initialize with all potential keys from get_all_possible_columns()
        # For minimal change, initialize empty and add as found.
        data = {}

        try:
            # Rate limiting (Keep your original logic here)
            now = time.time()
            if hasattr(self, '_last_request_time') and self._last_request_time > 0:  # Check > 0
                elapsed = now - self._last_request_time
                # Use delay from params if available, else a default
                required_delay = self.params.delay_between_batches if self.params else 1.0
                if elapsed < required_delay:
                    sleep_time = required_delay - elapsed
                    logging.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
            self._last_request_time = time.time()  # Update time *after* potential sleep

            logging.debug(f"Requesting URL: {url}")
            self.driver.get(url)
            # Use hover_wait from params if available, else a default
            time.sleep(self.params.hover_wait if self.params else 0.5)

            # --- Attempt to switch to CVSS 3.x tab ---
            self._switch_to_cvss3()  # Keep this call

            # --- CORE CHANGE: Extract metrics from hidden inputs ---
            # Iterate through the configured sources (cna, nist, adp)
            for source in self.HIDDEN_INPUT_IDS.keys():
                # Call the method that parses the hidden input
                source_data = self._extract_from_hidden_input(source)
                # Update the main data dict with prefixed keys (e.g., cna_base_score_num)
                data.update(source_data)
            # --- END CORE CHANGE ---

            # --- Extract base properties (non-metrics) using direct XPath ---
            # Keep using extract_property for these, assuming they still work
            base_properties_to_extract = ['nvd_published_date', 'nvd_description']
            for prop in base_properties_to_extract:
                # Check if this property should be extracted based on initialization list
                # (Optional check, original code didn't show this check here)
                # if prop in self.properties_to_extract:
                value = self.extract_property(prop)
                if value is not None:
                    data[prop] = value
                # else: data[prop] will be missing or None if initialized

            # --- Extract CWE data ---
            # Keep using extract_cwe_data, assuming it works
            # Check if any CWE property should be extracted
            if any(p.startswith('cwe_') for p in self.properties_to_extract):
                cwe_data = self.extract_cwe_data()
                if cwe_data:
                    # Aggregate multi-CWE entries using '; ' separator
                    # Check individual properties before joining
                    if 'cwe_id' in self.properties_to_extract:
                        data['cwe_id'] = '; '.join(filter(None, [entry.get('cwe_id') for entry in cwe_data]))
                    if 'cwe_name' in self.properties_to_extract:
                        data['cwe_name'] = '; '.join(filter(None, [entry.get('cwe_name') for entry in cwe_data]))
                    if 'cwe_source' in self.properties_to_extract:
                        data['cwe_source'] = '; '.join(filter(None, [entry.get('cwe_source') for entry in cwe_data]))
                    if 'cwe_url' in self.properties_to_extract:
                        data['cwe_url'] = '; '.join(filter(None, [str(entry.get('cwe_url')) for entry in cwe_data if entry.get('cwe_url')]))
                else:
                    # Ensure CWE fields are None if no data found but they were requested
                    if 'cwe_id' in self.properties_to_extract: data['cwe_id'] = None  # noqa: E701
                    if 'cwe_name' in self.properties_to_extract: data['cwe_name'] = None  # noqa: E701
                    if 'cwe_source' in self.properties_to_extract: data['cwe_source'] = None  # noqa: E701
                    if 'cwe_url' in self.properties_to_extract: data['cwe_url'] = None  # noqa: E701

        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}", exc_info=True)  # Add exc_info for traceback
            # Ensure all *requested* properties exist in the dict, even if None, upon error
            # for prop in self.properties_to_extract:
            #    if prop not in data:
            #        data[prop] = None
            # Simpler: just return potentially incomplete data, caller handles it.

        # Filter data to return only the properties requested during initialization
        # This respects the original 'properties_to_extract' logic
        final_data = {}
        # Explicitly log the list of properties we are expecting to include
        logging.debug(f"Constructing final_data based on self.properties_to_extract: {self.properties_to_extract}")

        # Iterate through the definitive list of properties that *should* be extracted
        for prop_key in self.properties_to_extract:
            # Get the value from the intermediate 'data' dict (which holds all scraped values)
            # Use .get() to safely handle cases where a key might be missing from 'data' (though less likely now)
            value = data.get(prop_key)
            final_data[prop_key] = value

            # Add specific debug logging for the missing keys
            if prop_key.endswith(('_base_score_num', '_base_score_rating')):
                logging.debug(f"Final Check: Assigning '{prop_key}' with value '{value}' to final_data.")

        # Log the completed final_data for comparison
        logging.debug(f"Returning final_data: {final_data}")
        return final_data

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        url_column: str = 'post_id',
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        Augments the input DataFrame with NVD data scraped from URLs.
        Handles initialization, scraping loop, data assignment, and NaN normalization.

        Args:
            df (pd.DataFrame): Input DataFrame subset needing NVD data.
            url_column (str): Column name containing the CVE ID.
            batch_size (int): Controls loop chunking (less relevant now but kept).

        Returns:
            pd.DataFrame: DataFrame with NVD columns populated (or None if errors).
                         NaN values are normalized to Python None.
        """
        if self.driver is None:
            logging.error("WebDriver is not initialized. Cannot augment DataFrame.")
            # Return a copy of the input df structure but empty, or the original?
            # Returning original might be misleading. Let's return an empty copy with NVD cols.
            df_copy = df.copy()
            # Add NVD columns if they don't exist
            all_nvd_columns = self.get_all_possible_columns()
            for col in all_nvd_columns:
                if col not in df_copy.columns:
                    df_copy[col] = None
            return df_copy

        # Work on a copy of the input DataFrame (which should be the subset)
        df_copy = df.copy()
        total_rows = len(df_copy)  # Process all rows passed to this function
        logging.debug(f"Augmenting NVD data for {total_rows} rows.")

        # 1. Initialize all potential NVD output columns if they don't exist
        all_nvd_columns = self.get_all_possible_columns()
        initialized_cols_count = 0
        for col in all_nvd_columns:
            if col not in df_copy.columns:
                # Initialize with None - allows mixed types initially
                df_copy[col] = None
                initialized_cols_count += 1
        if initialized_cols_count > 0:
            logging.debug(f"Initialized {initialized_cols_count} missing NVD columns with None.")

        # Determine rows to process (in this context, all rows passed in df_copy)
        rows_to_process_indices = df_copy.index

        # Progress bar setup
        progress_bar = (
            tqdm(total=total_rows, desc="Scraping NVD Data")
            if self.show_progress
            else None
        )

        # 2. Process rows needing NVD data
        try:
            # Loop through the indices of the subset DataFrame passed in
            for current_iter, idx in enumerate(rows_to_process_indices):
                row = df_copy.loc[idx]
                post_id = row.get(url_column)

                # --- Input Validation ---
                if pd.isna(post_id) or not str(post_id).strip():
                    logging.warning(f"Skipping row index {idx} due to missing/invalid CVE ID.")
                    if progress_bar: progress_bar.update(1)  # noqa: E701
                    continue
                post_id_str = str(post_id).strip().upper()
                if not post_id_str.startswith("CVE-"):
                    logging.warning(f"Row index {idx}: Value '{post_id}' in '{url_column}' is not a valid CVE ID. Skipping.")
                    if progress_bar: progress_bar.update(1)  # noqa: E701
                    continue
                # --- End Input Validation ---

                url = f"https://nvd.nist.gov/vuln/detail/{post_id_str}"
                extracted_data = {}  # Initialize for this row

                try:
                    # --- Call the scraping function ---
                    extracted_data = self.extract_data_from_url(url)

                    # --- *** CRITICAL DEBUGGING STEP *** ---
                    if current_iter < 5:  # Log the first few results
                        logging.debug(f"Extracted data for {post_id_str} (Index {idx}): {extracted_data}")
                    # --- *** END DEBUGGING STEP *** ---

                    # --- Assign scraped data to the DataFrame copy ---
                    if extracted_data:  # Check if *anything* was returned
                        assigned_count = 0
                        for key, value in extracted_data.items():
                            if key in df_copy.columns:
                                # Use .loc for assignment to avoid SettingWithCopyWarning
                                df_copy.loc[idx, key] = value
                                assigned_count += 1
                            else:
                                logging.warning(f"Skipping assignment for key '{key}' - column not found in DataFrame copy.")
                        if assigned_count == 0 and extracted_data:
                            logging.warning(f"Extracted data for {post_id_str} but assigned 0 values (check column names).")
                    else:
                        logging.warning(f"No data extracted for {post_id_str} (Index {idx}). Row will have None values.")
                        # Ensure NVD columns are None for this row if extraction failed completely
                        for col in all_nvd_columns:
                            if col in df_copy.columns:
                                df_copy.loc[idx, col] = None

                except Exception as e:
                    logging.error(f"Failed during scraping or assignment for index {idx} (CVE: {post_id_str}): {e}", exc_info=True)
                    # Ensure NVD columns are None for this row on error
                    for col in all_nvd_columns:
                        if col in df_copy.columns:
                            df_copy.loc[idx, col] = None

                finally:
                    if progress_bar:
                        progress_bar.update(1)

        finally:
            if progress_bar:
                progress_bar.close()

        # 3. --- Normalize NaN to None ---
        # Before setting final types, replace pandas NaN with Python None for consistency
        logging.debug("Normalizing NaN values to None before setting final types.")
        # Create a copy to avoid SettingWithCopyWarning during replace
        df_copy_normalized = df_copy.copy()
        nvd_cols_in_df = [col for col in all_nvd_columns if col in df_copy_normalized.columns]
        for col in nvd_cols_in_df:
            # Replace pd.NA and np.nan if they exist. astype(object) helps if column is numeric.
            try:
                # Convert potential pd.NA (nullable int/float) to np.nan first, then replace np.nan with None
                if pd.api.types.is_numeric_dtype(df_copy_normalized[col]):
                    # Use fillna(np.nan) then replace might be safer for specific numeric types
                    df_copy_normalized[col] = df_copy_normalized[col].replace({pd.NA: None})  # Replace pandas NA first if nullable type
                    df_copy_normalized[col] = df_copy_normalized[col].replace({np.nan: None})  # Replace numpy NaN
                else:
                    # For object/string types, replace np.nan just in case
                    df_copy_normalized[col] = df_copy_normalized[col].replace({np.nan: None})

                # Explicitly replace string 'None'/'nan' if they accidentally crept in (less likely now)
                if pd.api.types.is_string_dtype(df_copy_normalized[col]) or df_copy_normalized[col].dtype == 'object':
                    df_copy_normalized[col] = df_copy_normalized[col].replace({'None': None, 'nan': None, 'NaN': None})

            except Exception as e:
                logging.warning(f"Could not normalize NaN/None in column '{col}': {e}")

        # 4. Apply final column types (optional here, could be done later)
        # If you keep it here, ensure _set_column_types handles None correctly
        logging.debug("Applying final column types.")
        try:
            df_final = self._set_column_types(df_copy_normalized)
            # One final check/replace after type setting if _set_column_types reintroduces NaN
            # for col in nvd_cols_in_df:
            #     if pd.api.types.is_numeric_dtype(df_final[col]):
            #          df_final[col] = df_final[col].replace({np.nan: None})

        except Exception as e:
            logging.error(f"Error applying column types: {e}. Returning DataFrame with normalized None values but possibly incorrect dtypes.", exc_info=True)
            df_final = df_copy_normalized  # Return the normalized but untyped DF

        logging.debug(f"Finished augmenting NVD data. Returning {len(df_final)} rows.")
        return df_final

    def get_output_columns(self) -> List[str]:
        """Get list of all column names that will be added by this extractor.

        Returns:
            List[str]: List of column names that will be added to the DataFrame.
        """
        columns = []

        # Add base properties that are always included
        base_properties = ['nvd_published_date', 'nvd_description']
        columns.extend(base_properties)

        # Add vector metrics with source prefixes
        for source in self.VECTOR_SOURCES.keys():  # nist, cna, adp
            prefix = self.VECTOR_SOURCES[source]['prefix']  # nist_, cna_, adp_

            # Add base score and vector for each source
            columns.append(f"{prefix}vector")
            columns.append(f"{prefix}base_score_num")
            columns.append(f"{prefix}base_score_rating")
            for metric in self.METRIC_PATTERNS.keys():
                columns.append(f"{prefix}{metric}")

        return sorted(list(set(columns)))

    @staticmethod
    def get_all_possible_columns() -> List[str]:
        """Get list of ALL possible column names that could be added or populated."""
        columns = set()

        # Base properties
        base_properties = ['nvd_published_date', 'nvd_description']
        columns.update(base_properties)

        # CWE properties
        cwe_columns = ['cwe_id', 'cwe_name', 'cwe_source', 'cwe_url']
        columns.update(cwe_columns)

        # Metrics from hidden inputs (prefixed)
        # Explicitly define the patterns for base score components
        base_score_keys = ['base_score', 'base_score_num', 'base_score_rating']
        # Get other metric keys directly from the dictionary keys
        other_metric_keys = [
            k for k in NVDDataExtractor.HIDDEN_INPUT_METRIC_TEST_IDS.keys()
            if k not in base_score_keys  # Avoid duplicating base score keys
        ]

        source_prefixes = [info['prefix'] for info in NVDDataExtractor.VECTOR_SOURCES.values()]  # ['nist_', 'cna_', 'adp_']

        for prefix in source_prefixes:
            # Add base score keys explicitly
            for key in base_score_keys:
                columns.add(f"{prefix}{key}")
            # Add other metric keys derived from the test ID dictionary
            for key in other_metric_keys:
                columns.add(f"{prefix}{key}")

        logging.debug(f"Generated all possible columns: {sorted(list(columns))}")
        return sorted(list(columns))

    def debug_tooltip_content(self, tooltip_element) -> None:
        """Debug helper to print the tooltip's HTML content.

        Args:
            tooltip_element: Tooltip element to extract content from.
        """
        try:
            html_content = tooltip_element.get_attribute('innerHTML')
            logging.debug(f"Tooltip HTML content:\n{html_content}")
        except Exception as e:
            logging.error(f"Failed to get tooltip content: {str(e)}")


def main() -> None:
    """Main function for testing the NVD Data Extractor.

    Tests the extraction of CVSS metrics from multiple NVD vulnerability pages,
    verifying extraction from different sources (CNA, NIST, ADP).
    """
    # Test URLs for different metric sources
    test_urls = [
        {
            "url": "https://nvd.nist.gov/vuln/detail/CVE-2025-21229",
            "description": "CVE with CNA metrics"
        },
        {
            "url": "https://nvd.nist.gov/vuln/detail/CVE-2023-5475",
            "description": "CVE with NIST metrics"
        },
        {
            "url": "https://nvd.nist.gov/vuln/detail/CVE-2024-50660",
            "description": "CVE with ADP metrics"
        }
    ]

    # Initialize extractor with debug logging
    extractor = NVDDataExtractor(
        headless=False,  # Set to False to see the browser
        window_size=(1920, 1080),
        show_progress=True
    )

    try:
        for test_case in test_urls:
            print(f"\n\nTesting {test_case['description']}:")
            print(f"URL: {test_case['url']}")
            print("-" * 80)

            # Navigate to the test URL
            extractor.driver.get(test_case['url'])
            time.sleep(2)  # Wait for page load

            # Extract vector metrics
            metrics_data = extractor.extract_vector_metrics()

            # Print extracted data
            print("\nExtracted Vector Metrics:")
            print(json.dumps(metrics_data, indent=2))

            # Extract CWE data
            cwe_data = extractor.extract_cwe_data()
            print("\nExtracted CWE Data:")
            print(json.dumps(cwe_data, indent=2))

            # Add a separator between test cases
            print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error during extraction: {str(e)}")
    finally:
        # Clean up
        if hasattr(extractor, 'driver'):
            extractor.driver.quit()


if __name__ == "__main__":
    main()
