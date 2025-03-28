The NVD display page hides the tooltip content when the page is loaded.

The tooltip content is only visible after a user hovers over the element.

The html in question, looks like the following:

```html
<span data-testid="vuln-cvss3-cna-vector" class="tooltipCvss3CnaMetrics" data-original-title="" title="" aria-describedby="tooltip879274">CVSS:3.1/AV:P/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H</span>

<div class="tooltip fade bottom in" role="tooltip" id="tooltip829763" style="top: 23px; left: 86.4375px; display: block;"><div class="tooltip-arrow" style="left: 50%;"></div><div class="tooltip-inner"><span id="nistV3Metric"> <strong> CVSS v3.1 Severity and Metrics:</strong> <p> <strong>Base Score: </strong> <span>6.6 </span><span>MEDIUM</span> <br> <strong>Vector: </strong> <span> AV:P/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H </span> <br> <strong>Impact Score: </strong> <span> 5.9 </span> <br> <strong>Exploitability Score: </strong> <span> 0.7 </span> </p> <hr> <p> <strong>Attack Vector (AV): </strong> <span> Physical </span> <br> <strong>Attack Complexity (AC): </strong> <span> Low </span> <br> <strong>Privileges Required (PR): </strong> <span> Low </span> <br> <strong>User Interaction (UI): </strong> <span> None </span> <br> <strong>Scope (S): </strong> <span> Unchanged </span> <br> <strong>Confidentiality (C): </strong> <span> High </span> <br> <strong>Integrity (I): </strong> <span> High </span> <br> <strong>Availability (A): </strong> <span> High </span> </p> </span> </div></div>

<input type="hidden" id="cnaV3MetricHidden" value="<span id = 'nistV3Metric' style = 'display:none' > <strong style ='font-size:1.2em' > CVSS v3.1 Severity and Metrics:</strong> <p data-testid='vuln-cvssv3-score-container'> <strong>Base Score: </strong> <span data-testid='vuln-cvssv3-base-score'>6.6 </span><span data-testid='vuln-cvssv3-base-score-severity'>MEDIUM</span> <br /> <strong>Vector: </strong> <span data-testid='vuln-cvssv3-vector'> AV:P/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H </span> <br /> <strong>Impact Score: </strong> <span data-testid='vuln-cvssv3-impact-score'> 5.9 </span> <br /> <strong>Exploitability Score: </strong> <span data-testid='vuln-cvssv3-exploitability-score'> 0.7 </span> </p> <hr /> <p data-testid='vuln-cvssv3-metrics-container'> <strong>Attack Vector (AV): </strong> <span data-testid='vuln-cvssv3-av'> Physical </span> <br /> <strong>Attack Complexity (AC): </strong> <span data-testid='vuln-cvssv3-ac'> Low </span> <br /> <strong>Privileges Required (PR): </strong> <span data-testid='vuln-cvssv3-pr'> Low </span> <br /> <strong>User Interaction (UI): </strong> <span data-testid='vuln-cvssv3-ui'> None </span> <br /> <strong>Scope (S): </strong> <span data-testid='vuln-cvssv3-s'> Unchanged </span> <br /> <strong>Confidentiality (C): </strong> <span data-testid='vuln-cvssv3-c'> High </span> <br /> <strong>Integrity (I): </strong> <span data-testid='vuln-cvssv3-i'> High </span> <br /> <strong>Availability (A): </strong> <span data-testid='vuln-cvssv3-a'> High </span> </p> </span> ">
```

Below is an example of how you can minimally modify your existing NVDDataExtractor class to fallback to parsing the hidden CNA <input> (rather than relying on hover) when the tooltip-based extraction fails. The changes are:

Add a small private method, _extract_cna_from_hidden_input, which:

Locates the cnaV3MetricHidden <input> field.

Extracts its value attribute.

Parses the embedded HTML for CVSS data (using BeautifulSoup or a simple Selenium-based string search).

Update the extract_vector_metrics method to fallback to _extract_cna_from_hidden_input when the CNA tooltip approach fails (e.g., element not interactable).

Everything else remains unchanged, preserving the existing logic for nist and adp.

```python

def _extract_cna_from_hidden_input(self) -> Dict[str, Any]:
    """
    Fallback approach for CNA metrics:
    Scrape the hidden <input> with id="cnaV3MetricHidden"
    instead of relying on a hover tooltip.
    """
    data = {
        "cna_vector": None,
        "cna_base_score_num": None,
        "cna_base_score_rating": None,
        "cna_impact_score": None,
        "cna_exploitability_score": None,
        # ... add any other cna_ prefixed fields you need ...
    }

    try:
        # 1) Locate hidden <input> by ID
        cna_input = self.driver.find_element(By.ID, "cnaV3MetricHidden")
        raw_html = cna_input.get_attribute("value")
        if not raw_html:
            logging.debug("cnaV3MetricHidden found but empty.")
            return data  # All fields stay None

        # 2) Parse the embedded HTML
        soup = BeautifulSoup(raw_html, "html.parser")

        # -- Example: Base Score numeric value --
        base_score_tag = soup.find("span", {"data-testid": "vuln-cvssv3-base-score"})
        if base_score_tag:
            data["cna_base_score_num"] = convert_to_float(base_score_tag.text.strip())

        # -- Example: Base Score severity --
        severity_tag = soup.find("span", {"data-testid": "vuln-cvssv3-base-score-severity"})
        if severity_tag:
            data["cna_base_score_rating"] = severity_tag.text.strip().lower()

        # -- Example: Vector --
        vector_tag = soup.find("span", {"data-testid": "vuln-cvssv3-vector"})
        if vector_tag:
            data["cna_vector"] = vector_tag.text.strip()

        # -- Impact & Exploitability Scores --
        impact_tag = soup.find("span", {"data-testid": "vuln-cvssv3-impact-score"})
        if impact_tag:
            data["cna_impact_score"] = convert_to_float(impact_tag.text.strip())

        exploitability_tag = soup.find("span", {"data-testid": "vuln-cvssv3-exploitability-score"})
        if exploitability_tag:
            data["cna_exploitability_score"] = convert_to_float(exploitability_tag.text.strip())

        # ... replicate for Attack Vector, Complexity, etc. as needed ...
        logging.debug("Successfully parsed CNA metrics from hidden input.")

    except NoSuchElementException:
        logging.debug("No hidden CNA input found on this page.")
    except Exception as e:
        logging.error(f"Error extracting CNA metrics from hidden input: {str(e)}")

    return data

def extract_vector_metrics(self) -> Dict[str, Any]:
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
            # == Existing logic: find vector, hover, parse tooltip, etc. ==
            vector_element = self.driver.find_element(
                By.XPATH, selectors['vector_element']
            )
            vector_text = vector_element.get_attribute('textContent').strip()
            data[f"{prefix}vector"] = vector_text

            ActionChains(self.driver).move_to_element(vector_element).perform()
            time.sleep(self.params.hover_wait)

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
                    # (existing numeric handling, etc.)
                    ...
                except NoSuchElementException:
                    logging.warning(f"No element found for {metric}")
                    continue
                except Exception as ex:
                    logging.error(f"Error extracting {metric}: {str(ex)}")
                    continue

        except NoSuchElementException:
            logging.debug(f"No vector element found for {source}")
            continue
        except Exception as e:
            # Fallback if source == 'cna'
            logging.error(f"Error processing {source} metrics: {str(e)}")
            if source == 'cna':
                logging.info("Falling back to cna hidden input approach.")
                cna_data = self._extract_cna_from_hidden_input()
                # Merge the fallback data into main dictionary
                data.update(cna_data)
            continue

    logging.debug(f"\nExtracted data: {json.dumps(data, indent=2)}")
    return data
```
