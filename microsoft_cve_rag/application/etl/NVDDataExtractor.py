import os
import time
import pandas as pd
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, InvalidSelectorException
from selenium.webdriver.common.action_chains import ActionChains
from tqdm import tqdm
from functools import wraps
from dataclasses import dataclass
import math
from typing import List, Dict, Optional, Any, Tuple
import json
from application.app_utils import setup_logger
from application.etl.type_utils import convert_to_float

logging.getLogger(__name__)


@dataclass
class ScrapingParams:
    """Parameters for controlling scraping speed and behavior"""
    implicit_wait: float  # seconds to wait for elements
    hover_wait: float    # seconds to wait for tooltips
    page_load_wait: float  # seconds to wait for page load
    rate_limit: float    # minimum seconds between requests

    @classmethod
    def from_target_time(cls, num_cves: int, target_time_per_cve: float = 10.0) -> 'ScrapingParams':
        """
        Calculate optimal scraping parameters based on desired processing time.
        Uses a logarithmic scaling to increase delays as batch size grows.
        """
        base_implicit_wait = 2.0
        base_hover_wait = 0.5
        base_page_load_wait = 1.0
        base_rate_limit = 1.0

        scale = math.log2(max(num_cves, 2)) / 4

        return cls(
            implicit_wait=min(base_implicit_wait * scale, 4.0),
            hover_wait=min(base_hover_wait * scale, 1.0),
            page_load_wait=min(base_page_load_wait * scale, 2.0),
            rate_limit=min(base_rate_limit * scale, 3.0)
        )

    def estimate_time_per_cve(self) -> float:
        base_time = self.implicit_wait + self.hover_wait + self.page_load_wait + self.rate_limit
        selenium_overhead = 1.0
        network_latency = 2.0
        parsing_time = 1.0
        tooltip_wait = 1.0

        return base_time + selenium_overhead + network_latency + parsing_time + tooltip_wait

    def estimate_total_time(self, num_cves: int) -> float:
        return self.estimate_time_per_cve() * num_cves

class NVDDataExtractor:
    # Element mappings as class constant
    ELEMENT_MAPPINGS = {
        'nvd_published_date': "//span[@data-testid='vuln-published-on']",
        'nvd_description': "//p[@data-testid='vuln-description']",
        'base_score_num': "//a[@data-testid='vuln-cvss3-panel-score']",
        'base_score_rating': "//a[@data-testid='vuln-cvss3-panel-score']",
        'vector_element': "//span[@data-testid='vuln-cvssv3-vector']",
        'cwe_id': "//td[contains(@data-testid, 'vuln-CWEs-link-')]",
        'cwe_name': "//td[contains(@data-testid, 'vuln-CWEs-link-')]/following-sibling::td[1]",
        'cwe_source': "//td[contains(@data-testid, 'vuln-CWEs-link-')]/following-sibling::td[2]"
    }
    VECTOR_SOURCES = {
        'nist': {
            'prefix': 'nist_',
            'vector_element': "//span[@data-testid='vuln-cvss3-nist-vector']",
            'base_score': "//span[@data-testid='vuln-cvss3-nist-panel-score']"
        },
        'cna': {
            'prefix': 'cna_',
            'vector_element': "//span[@data-testid='vuln-cvss3-cna-vector']",
            'base_score': "//span[@data-testid='vuln-cvss3-cna-panel-score']"
        },
        'adp': {
            'prefix': 'adp_',
            'vector_element': "//span[@data-testid='vuln-cvss3-adp-vector']",
            'base_score': "//span[@data-testid='vuln-cvss3-adp-panel-score']"
        }
    }
    METRIC_PATTERNS = {
        'base_score': ".//p/strong[contains(text(), 'Base Score:')]/following-sibling::span[1]",
        'base_score_rating': ".//p/strong[contains(text(), 'Base Score:')]/following-sibling::span[2]",
        'vector': ".//p/strong[contains(text(), 'Vector:')]/following-sibling::span",
        'impact_score': ".//p/strong[contains(text(), 'Impact Score:')]/following-sibling::span",
        'exploitability_score': ".//p/strong[contains(text(), 'Exploitability Score:')]/following-sibling::span",
        'attack_vector': ".//p/strong[contains(text(), 'Attack Vector')]/following-sibling::span",
        'attack_complexity': ".//p/strong[contains(text(), 'Attack Complexity')]/following-sibling::span",
        'privileges_required': ".//p/strong[contains(text(), 'Privileges Required')]/following-sibling::span",
        'user_interaction': ".//p/strong[contains(text(), 'User Interaction')]/following-sibling::span",
        'scope': ".//p/strong[contains(text(), 'Scope')]/following-sibling::span",
        'confidentiality': ".//p/strong[contains(text(), 'Confidentiality')]/following-sibling::span",
        'integrity': ".//p/strong[contains(text(), 'Integrity')]/following-sibling::span",
        'availability': ".//p/strong[contains(text(), 'Availability')]/following-sibling::span"
    }

    def __init__(
        self,
        properties_to_extract: Optional[List[str]] = None,
        max_records: Optional[int] = None,
        headless: bool = True,
        window_size: Optional[Tuple[int, int]] = None,
        scraping_params: Optional[ScrapingParams] = None,
        show_progress: bool = False
    ) -> None:
        """
        Initialize the NVDDataExtractor with specified properties and browser settings.

        Args:
            properties_to_extract: List of properties to extract from NVD pages.
            max_records: Maximum number of records to process. If None, process all records.
            headless: Whether to run Chrome in headless mode. Default True.
            window_size: Tuple of (width, height) for browser window.
            scraping_params: Optional custom scraping parameters
            show_progress: Whether to show progress bar. Default False.
        """
        self.valid_properties = (
            set(self.ELEMENT_MAPPINGS.keys()) |  # Base properties
            set(self.METRIC_PATTERNS.keys()) |    # Metric properties
            {'base_score', 'vector_element'} |    # Special properties
            {'cwe_id', 'cwe_name', 'cwe_source', 'cwe_url'}  # CWE properties
        )
        # Validate and normalize properties to extract
        if properties_to_extract:
            invalid_props = set(properties_to_extract) - self.valid_properties
            if invalid_props:
                logging.warning(f"Ignoring invalid properties: {invalid_props}")
            self.properties_to_extract = [p for p in properties_to_extract if p in self.valid_properties]
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
        chrome_options = webdriver.ChromeOptions()
        if self.headless:
            chrome_options.add_argument('--headless=new')
        chrome_options.add_argument(f'--window-size={self.window_size[0]},{self.window_size[1]}')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.page_load_strategy = 'eager'
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_argument('--silent')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])

        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(self.params.implicit_wait)

    def cleanup(self) -> None:
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logging.debug("Selenium WebDriver cleaned up.")
            except Exception as e:
                logging.error(f"Error during driver cleanup: {str(e)}")

    def _extract_base_score(self, score_text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract numeric score and rating from score text (e.g. '7.5 HIGH')"""
        try:
            if not score_text or score_text.lower() == 'none':
                return None, None

            parts = score_text.strip().split(' ', 1)
            score = float(parts[0]) if parts[0] else None
            rating = parts[1].lower() if len(parts) > 1 else 'none'
            return score, rating
        except (ValueError, IndexError) as e:
            logging.warning(f"Error parsing base score '{score_text}': {str(e)}")
            return None, 'none'

    def _set_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sets appropriate data types for all columns in the DataFrame.
        Ensures datetime fields are Python datetime objects for Neomodel compatibility.

        Args:
            df: Input DataFrame with raw column types

        Returns:
            pd.DataFrame: DataFrame with correct column types
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
                    return datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S')
                except ValueError as e:
                    logging.warning(f"Could not parse NVD date: {date_str} - {str(e)}")
                    return None
        # Apply source prefixes to relevant columns
        prefixed_types = {}
        for source in self.VECTOR_SOURCES.keys():
            prefix = self.VECTOR_SOURCES[source]['prefix']
            for col, dtype in type_mappings.items():
                if col in ['base_score_num', 'base_score_rating'] + list(self.METRIC_PATTERNS.keys()):
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
                    logging.warning(f"Failed to convert column {column} to {dtype}: {str(e)}")

        return df

    def extract_property(self, property_name: str, context_element: Optional[Any] = None) -> Optional[str]:
        """
        Extract a single property from the page using predefined element mappings.

        Args:
            property_name: Name of the property to extract
            context_element: Parent element to search within (optional)

        Returns:
            Optional[str]: The extracted text value or None if not found
        """
        if property_name not in self.ELEMENT_MAPPINGS:
            print(f"property name: {property_name}")
            logging.warning(f"No mapping found for property: {property_name}")
            return 'none'

        try:
            element = self.driver.find_element(
                By.XPATH,
                self.ELEMENT_MAPPINGS[property_name]
            )
            return element.text.lower().strip()
        except NoSuchElementException:
            logging.debug(f"Element not found for property: {property_name}")
            return 'none'
        except Exception as e:
            logging.warning(f"Error extracting {property_name}: {str(e)}")
            return 'none'

    def extract_vector_metrics(self) -> Dict[str, Optional[str]]:
        """Extract all metrics from tooltip for each source"""
        data = {}

        for source, selectors in self.VECTOR_SOURCES.items():
            prefix = selectors['prefix']
            logging.debug(f"\nProcessing {source} metrics...")

            # Initialize all fields for this source as None
            data[f"{prefix}vector"] = None
            data[f"{prefix}base_score_num"] = None
            data[f"{prefix}base_score_rating"] = None
            for metric in self.METRIC_PATTERNS.keys():
                data[f"{prefix}{metric}"] = None

            try:
                # Find vector element
                vector_element = self.driver.find_element(
                    By.XPATH,
                    selectors['vector_element']
                )

                # Get vector string directly from the element
                vector_text = vector_element.get_attribute('textContent').strip()
                data[f"{prefix}vector"] = vector_text
                logging.debug(f"Found vector: {vector_text}")

                # Hover to show tooltip
                ActionChains(self.driver).move_to_element(vector_element).perform()
                time.sleep(self.params.hover_wait)

                # Find tooltip by ID (it will be dynamic)
                tooltip_id = vector_element.get_attribute('aria-describedby')
                if not tooltip_id:
                    logging.debug(f"No tooltip ID found for {source}")
                    continue

                tooltip = self.driver.find_element(By.ID, tooltip_id)

                # Extract all metrics from tooltip
                for metric, xpath in self.METRIC_PATTERNS.items():
                    try:
                        element = tooltip.find_element(By.XPATH, xpath)
                        value = element.text.strip()

                        # Special handling for numeric metric values
                        if metric in ['base_score', 'impact_score', 'exploitability_score']:
                            score = convert_to_float(value)
                            data[f"{prefix}{metric}"] = score

                            # Additional handling for base_score
                            if metric == 'base_score':
                                data[f"{prefix}base_score_num"] = score
                        elif metric == 'base_score_rating':
                            data[f"{prefix}base_score_rating"] = value.lower()
                        else:
                            data[f"{prefix}{metric}"] = value.lower()

                        logging.debug(f"Found {metric}: {value.lower()}")
                    except NoSuchElementException:
                        logging.warning(f"No element found for {metric}")
                        continue
                    except Exception as e:
                        logging.error(f"Error extracting {metric}: {str(e)}")
                        continue

            except NoSuchElementException:
                logging.debug(f"No vector element found for {source}")
                continue
            except Exception as e:
                logging.error(f"Error processing {source} metrics: {str(e)}")
                continue

        logging.debug(f"\nExtracted data: {json.dumps(data, indent=2)}")
        return data

    def extract_cwe_data(self) -> List[Dict[str, str]]:
        """
        Extracts CWE information from the Weakness Enumeration table if it exists.
        """
        cwe_data = []
        try:
            # Find the CWE table
            table = self.driver.find_element(
                By.XPATH,
                "//table[@data-testid='vuln-CWEs-table']"
            )

            # Find all rows except header
            rows = table.find_elements(
                By.XPATH,
                ".//tbody/tr"
            )

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
                        'cwe_source': source_cell.text.strip()
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

    def extract_data_from_url(self, url: str) -> Dict[str, Optional[str]]:
        """Extract data from the given URL based on the properties specified during initialization."""
        data = {}
        try:
            # Rate limiting and page load
            now = time.time()
            if hasattr(self, '_last_request_time'):
                elapsed = now - self._last_request_time
                if elapsed < self.params.rate_limit:
                    time.sleep(self.params.rate_limit - elapsed)
            self._last_request_time = time.time()

            self.driver.get(url)
            time.sleep(self.params.page_load_wait)

            # Extract vector metrics if requested (always include base_score)
            metric_properties = [p for p in self.properties_to_extract
                            if p in self.METRIC_PATTERNS or p in ['base_score', 'vector_element']]
            if metric_properties:
                vector_data = self.extract_vector_metrics()
                data.update(vector_data)

            # Extract base properties (nvd_published_date, nvd_description)
            base_properties = [p for p in self.properties_to_extract if p in self.ELEMENT_MAPPINGS]
            for prop in base_properties:
                try:
                    value = self.extract_property(prop)
                    if value is not None:
                        data[prop] = value
                except Exception as e:
                    logging.warning(f"Failed to extract {prop}: {str(e)}")
                    data[prop] = None

            # Extract CWE data if requested
            if any(prop.startswith('cwe_') for prop in self.properties_to_extract):
                cwe_data = self.extract_cwe_data()
                if cwe_data:
                    data.update({
                        'cwe_id': '; '.join(entry['cwe_id'] for entry in cwe_data),
                        'cwe_name': '; '.join(entry['cwe_name'] for entry in cwe_data),
                        'cwe_source': '; '.join(entry['cwe_source'] for entry in cwe_data),
                        'cwe_url': '; '.join(str(entry['cwe_url']) for entry in cwe_data if entry['cwe_url'])
                    })
                else:
                    data.update({
                        'cwe_id': None,
                        'cwe_name': None,
                        'cwe_source': None,
                        'cwe_url': None
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
        batch_size: int = 100
    ) -> pd.DataFrame:
        """Augments the given DataFrame with additional columns based on the extracted data."""
        df_copy = df.copy()
        total_rows = min(len(df_copy), self.max_records) if self.max_records else len(df_copy)

        # Initialize all possible columns
        columns_to_add = set()

        # Add source-prefixed columns for each source
        for source in self.VECTOR_SOURCES.keys():
            prefix = self.VECTOR_SOURCES[source]['prefix']
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
        progress_bar = tqdm(total=total_rows, desc="Processing CVEs") if self.show_progress else None

        try:
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch = df_copy.iloc[start_idx:end_idx]

                for idx, row in batch.iterrows():
                    # Skip if NVD data has already been extracted
                    if isinstance(row.get('metadata', {}), dict):
                        etl_status = row['metadata'].get('etl_processing_status', {})
                        if isinstance(etl_status, dict) and etl_status.get('nvd_extracted', False):
                            if progress_bar:
                                progress_bar.update(1)
                            logging.info(f"Skipping NVD extraction for {row[url_column]} - already extracted")
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
                            logging.debug(f"Skipping column {key} - not in initialized columns")

                    if progress_bar:
                        progress_bar.update(1)

                if self.max_records and end_idx >= self.max_records:
                    break

        finally:
            if progress_bar:
                progress_bar.close()

        return self._set_column_types(df_copy)

    def get_output_columns(self) -> List[str]:
        """
        Returns a list of all column names that will be added by this extractor
        based on the properties_to_extract configuration.

        Returns:
            List[str]: List of column names that will be added to the DataFrame

        Implementation Notes:
            - columns = extractor.get_output_columns()
            - print("Columns that will be added:", columns)
        """
        columns = []

        # Add base properties that are always included
        base_properties = ['nvd_published_date', 'description']
        columns.extend(base_properties)

        # Add vector metrics with source prefixes
        for source in self.VECTOR_SOURCES.keys():  # nist, cna, adp
            prefix = self.VECTOR_SOURCES[source]['prefix']  # nist_, cna_, adp_

            # Add base score and vector for each source
            columns.append(f"{prefix}base_score")
            columns.append(f"{prefix}vector")

            # Add all requested vector metrics with source prefix
            for prop in self.properties_to_extract:
                if prop in self.METRIC_PATTERNS:
                    columns.append(f"{prefix}{prop}")

        return sorted(list(set(columns)))

    @staticmethod
    def get_all_possible_columns() -> List[str]:
        """
        Returns a list of all possible column names that could be added by this extractor,
        regardless of the properties_to_extract configuration.

        Returns:
            List[str]: Complete list of all possible column names

        Example:
            columns = NVDDataExtractor.get_all_possible_columns()
            print("All possible columns:", columns)
        """
        columns = []

        # Base properties that are always included
        base_properties = ['nvd_published_date', 'description']
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
            'availability'
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
        """
        Debug helper to print the tooltip's HTML content
        """
        try:
            html_content = tooltip_element.get_attribute('innerHTML')
            logging.debug(f"Tooltip HTML content:\n{html_content}")
        except Exception as e:
            logging.error(f"Failed to get tooltip content: {str(e)}")



def main() -> None:
    """
    Main function to read the input CSV, extract data from NVD pages, and save the enriched DataFrame.
    """
    input_csv_path = r"C:\Users\emili\Downloads\Master_CVE_Information_Table_July-October_2024.csv"
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)

    # Initialize extractor with all properties you want to extract
    extractor = NVDDataExtractor(
        properties_to_extract=[
            'base_score',  # Will become nist_base_score and cna_base_score
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
            'nvd_published_date',
            'description',
            'vector_element'
        ],
        max_records=3,
    )
    NVDDataExtractor.get_all_possible_columns()
    # Process the DataFrame with batch size of 100 (you can adjust this)
    enriched_df = extractor.augment_dataframe(
        df=df,
        url_column='CVE ID',
        batch_size=100
    )

    output_csv_path = r"C:\Users\emili\Downloads\test_enriched_vulnerabilities.csv"
    enriched_df.to_csv(output_csv_path, index=False)

    processed_records = min(len(df), extractor.max_records) if extractor.max_records else len(df)
    summary = f"\nProcessing Summary:\n- Total rows processed: {processed_records}\n"
    logging.info(summary)

    extractor.cleanup()

if __name__ == "__main__":
    main()
