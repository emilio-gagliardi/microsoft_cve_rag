"""NVD Data Extractor for retrieving vulnerability information from the National Vulnerability Database.

This module provides functionality to extract detailed vulnerability information from
the NVD website using Selenium WebDriver. It handles the extraction of CVSS scores,
metrics, CWE data, and other vulnerability-related information.
"""

import json
import logging
import math
import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

from microsoft_cve_rag.application.etl.type_utils import convert_to_float

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
            'xpath': "//div[@data-testid='vuln-description']",
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

    def __init__(
        self,
        properties_to_extract: Optional[List[str]] = None,
        max_records: Optional[int] = None,
        headless: bool = True,
        window_size: Optional[Tuple[int, int]] = None,
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
        self.valid_properties = (
            set(self.ELEMENT_MAPPINGS.keys())  # Base properties
            | set(self.METRIC_PATTERNS.keys())  # Metric properties
            | {'base_score', 'vector_element'}  # Special properties
            | {'cwe_id', 'cwe_name', 'cwe_source', 'cwe_url'}  # CWE properties
        )
        # Validate and normalize properties to extract
        if properties_to_extract:
            invalid_props = set(properties_to_extract) - self.valid_properties
            if invalid_props:
                logging.warning(
                    f"Ignoring invalid properties: {invalid_props}"
                )
            self.properties_to_extract = [
                p for p in properties_to_extract if p in self.valid_properties
            ]
        else:
            self.properties_to_extract = list(self.valid_properties)
        self.max_records = max_records
        self.params = scraping_params or ScrapingParams.from_target_time(100)
        self.headless = headless
        self.window_size = window_size
        self.show_progress = show_progress
        self.driver = None
        self._last_request_time = 0
        self.setup_driver()

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
        """Extract metrics from hidden input field for a given source.

        Args:
            source: The source type ('cna', 'nist', or 'adp')

        Returns:
            Dict containing the extracted metrics with appropriate prefixes
        """
        prefix = self.VECTOR_SOURCES[source]['prefix']
        data = {
            f"{prefix}vector": None,
            f"{prefix}base_score_num": None,
            f"{prefix}base_score_rating": None,
            f"{prefix}impact_score": None,
            f"{prefix}exploitability_score": None
        }
        
        try:
            input_id = self.HIDDEN_INPUT_IDS[source]
            hidden_input = self.driver.find_element(By.ID, input_id)
            raw_html = hidden_input.get_attribute("value")
            
            if not raw_html:
                logging.debug(f"Hidden input for {source} found but empty")
                return data

            soup = BeautifulSoup(raw_html, "html.parser")
            
            # Extract base metrics
            base_score = soup.find("span", {"data-testid": "vuln-cvssv3-base-score"})
            if base_score:
                data[f"{prefix}base_score_num"] = convert_to_float(
                    base_score.text.strip()
                )
            
            severity = soup.find(
                "span", {"data-testid": "vuln-cvssv3-base-score-severity"}
            )
            if severity:
                data[f"{prefix}base_score_rating"] = severity.text.strip().lower()
            
            vector = soup.find("span", {"data-testid": "vuln-cvssv3-vector"})
            if vector:
                data[f"{prefix}vector"] = vector.text.strip()
            
            # Extract scores
            impact = soup.find("span", {"data-testid": "vuln-cvssv3-impact-score"})
            if impact:
                data[f"{prefix}impact_score"] = convert_to_float(
                    impact.text.strip()
                )
            
            exploit = soup.find(
                "span", {"data-testid": "vuln-cvssv3-exploitability-score"}
            )
            if exploit:
                data[f"{prefix}exploitability_score"] = convert_to_float(
                    exploit.text.strip()
                )
            
            logging.debug(f"Successfully parsed {source} metrics from hidden input")
            
        except NoSuchElementException:
            logging.debug(f"No hidden {source} input found")
        except Exception as e:
            logging.error(f"Error extracting {source} metrics from hidden input: {str(e)}")
        
        return data

    def _convert_to_float(self, value: str) -> Optional[float]:
        """Convert string to float, handling None and empty strings.

        Args:
            value: String value to convert

        Returns:
            Float value or None if conversion fails
        """
        try:
            return float(value.strip()) if value else None
        except (ValueError, AttributeError):
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
        """Extract data from the given URL based on the properties specified during initialization.

        Args:
            url (str): URL to extract data from.

        Returns:
            Dict[str, Any]: Dictionary containing extracted data.
        """
        data = {}
        try:
            # Rate limiting and page load
            now = time.time()
            if hasattr(self, '_last_request_time'):
                elapsed = now - self._last_request_time
                if elapsed < self.params.delay_between_batches:
                    time.sleep(self.params.delay_between_batches - elapsed)
            self._last_request_time = time.time()

            self.driver.get(url)
            time.sleep(self.params.hover_wait)

            # Switch to CVSS 3.x tab if needed
            self._switch_to_cvss3()

            # Extract vector metrics if requested (always include base_score)
            metric_properties = [
                p
                for p in self.properties_to_extract
                if p in self.METRIC_PATTERNS
                or p in ['base_score', 'vector_element']
            ]
            if metric_properties:
                vector_data = self.extract_vector_metrics()
                data.update(vector_data)

            # Extract base properties (nvd_published_date, nvd_description)
            base_properties = [
                p
                for p in self.properties_to_extract
                if p in self.ELEMENT_MAPPINGS
            ]
            for prop in base_properties:
                try:
                    value = self.extract_property(prop)
                    if value is not None:
                        data[prop] = value
                except Exception as e:
                    logging.warning(f"Failed to extract {prop}: {str(e)}")
                    data[prop] = None

            # Extract CWE data if requested
            if any(
                prop.startswith('cwe_') for prop in self.properties_to_extract
            ):
                cwe_data = self.extract_cwe_data()
                if cwe_data:
                    data.update({
                        'cwe_id': '; '.join(
                            entry['cwe_id'] for entry in cwe_data
                        ),
                        'cwe_name': '; '.join(
                            entry['cwe_name'] for entry in cwe_data
                        ),
                        'cwe_source': '; '.join(
                            entry['cwe_source'] for entry in cwe_data
                        ),
                        'cwe_url': '; '.join(
                            str(entry['cwe_url'])
                            for entry in cwe_data
                            if entry['cwe_url']
                        ),
                    })
                else:
                    data.update({
                        'cwe_id': None,
                        'cwe_name': None,
                        'cwe_source': None,
                        'cwe_url': None,
                    })

        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}")
            for prop in self.properties_to_extract:
                if prop not in data:
                    data[prop] = None

        return data

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        url_column: str = 'post_id',
        batch_size: int = 100,
    ) -> pd.DataFrame:
        """Augment DataFrame with additional columns based on extracted NVD data.

        Args:
            df (pd.DataFrame): Input DataFrame.
            url_column (str): Column containing NVD URLs.
            batch_size (int): Number of records to process in each batch.

        Returns:
            pd.DataFrame: Augmented DataFrame with additional NVD data columns.
        """
        df_copy = df.copy()
        total_rows = (
            min(len(df_copy), self.max_records)
            if self.max_records
            else len(df_copy)
        )

        # Initialize all possible columns
        columns_to_add = set()

        # Add source-prefixed columns for each source
        for source in self.VECTOR_SOURCES.keys():  # nist, cna, adp
            prefix = self.VECTOR_SOURCES[source]['prefix']  # nist_, cna_, adp_

            # Add base score and vector for each source
            columns_to_add.add(f"{prefix}vector")
            columns_to_add.add(f"{prefix}base_score_num")
            columns_to_add.add(f"{prefix}base_score_rating")
            for metric in self.METRIC_PATTERNS.keys():
                columns_to_add.add(f"{prefix}{metric}")

        # Add base columns
        columns_to_add.update(['nvd_published_date', 'nvd_description'])

        # Add CWE columns
        columns_to_add.update(['cwe_id', 'cwe_name', 'cwe_source', 'cwe_url'])

        # Initialize new columns with None
        for col in columns_to_add:
            if col not in df_copy.columns:
                df_copy[col] = None

        # Process rows
        progress_bar = (
            tqdm(total=total_rows, desc="Processing CVEs")
            if self.show_progress
            else None
        )

        try:
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch = df_copy.iloc[start_idx:end_idx]

                for idx, row in batch.iterrows():
                    # Skip if NVD data has already been extracted
                    if isinstance(row.get('metadata', {}), dict):
                        etl_status = row['metadata'].get(
                            'etl_processing_status', {}
                        )
                        if isinstance(etl_status, dict) and etl_status.get(
                            'nvd_extracted', False
                        ):
                            if progress_bar:
                                progress_bar.update(1)
                            logging.info(
                                "Skipping NVD extraction for"
                                f" {row[url_column]} - already extracted"
                            )
                            continue

                    post_id = row[url_column]
                    if pd.isna(post_id):
                        if progress_bar:
                            progress_bar.update(1)
                        continue

                    url = f"https://nvd.nist.gov/vuln/detail/{post_id}"
                    extracted_data = self.extract_data_from_url(url)

                    # Update DataFrame with extracted data
                    for key, value in extracted_data.items():
                        if key in df_copy.columns:
                            df_copy.at[idx, key] = value
                        else:
                            logging.debug(
                                f"Skipping column {key} - not in initialized"
                                " columns"
                            )

                    if progress_bar:
                        progress_bar.update(1)

                if self.max_records and end_idx >= self.max_records:
                    break

        finally:
            if progress_bar:
                progress_bar.close()

        return self._set_column_types(df_copy)

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
        """Get list of all possible column names that could be added by this extractor.

        Returns:
            List[str]: Complete list of all possible column names.
        """
        columns = []

        # Base properties that are always included
        base_properties = ['nvd_published_date', 'nvd_description']
        columns.extend(base_properties)

        # All possible vector metrics
        all_metrics = [
            'base_score',
            'vector',
            'impact_score',
            'exploitability_score',
            'attack_vector',
            'attack_complexity',
            'privileges_required',
            'user_interaction',
            'scope',
            'confidentiality',
            'integrity',
            'availability',
        ]
        cwe_columns = ['cwe_id', 'cwe_name', 'cwe_source', 'cwe_url']
        columns.extend(cwe_columns)
        # Add metrics for each source prefix (nist, cna, adp)
        source_prefixes = ['nist_', 'cna_', 'adp_']
        for prefix in source_prefixes:
            for metric in all_metrics:
                columns.append(f"{prefix}{metric}")

        return sorted(columns)

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
