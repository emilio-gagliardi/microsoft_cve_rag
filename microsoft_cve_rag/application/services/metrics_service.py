# microsoft_cve_rag/application/services/metrics_service.py

import asyncio
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import inspect  # To check if tables exist
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import (  # Optional: For more control over session creation
    sessionmaker,
)
from sqlmodel import Session, SQLModel, create_engine, select, text

# Import your shared models
try:
    from microsoft_cve_report_models.models import (
        DimMetric,
        DimTime,
        FactMetricValue,
    )
except ImportError:
    # Handle case where models might not be installed during certain tests/setups
    logging.error(
        "Could not import shared models from 'microsoft_cve_report_models'."
        " Ensure it's installed."
    )
    # Define dummy classes or raise a configuration error if necessary
    DimMetric = DimTime = FactMetricValue = None

logger = logging.getLogger(__name__)


class DuckDBMetricsService:
    """
    Provides an asynchronous interface to interact with the DuckDB metrics database
    using SQLModel. Designed for use within an async application like FastAPI.
    """

    def __init__(self, db_path: str | Path):
        """
        Initializes the service with the path to the DuckDB database file.

        Args:
            db_path: Path to the .db file (e.g., 'C:/path/to/data/metrics.db').
                     It's recommended to use an absolute path.
        """
        if not DimMetric:  # Check if models loaded
            raise RuntimeError(
                "Shared models could not be imported. Cannot initialize"
                " DuckDBMetricsService."
            )

        self.db_path = Path(db_path).resolve()
        logger.info(f"Metrics database path loaded: {self.db_path}")
        full_db_path = f"duckdb:///{self.db_path}"
        logger.info(f"Full metrics database path: {full_db_path}")
        self.database_url = full_db_path
        self._engine = None
        self._initialize_engine()
        # Optional: Create a sessionmaker if you want reusable session configurations
        # self._session_local = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)

        logger.info(
            f"DuckDBMetricsService initialized for database: {self.db_path}"
        )
        # Verify connection and table existence on init (optional but good)
        # asyncio.run(self.verify_connection_and_tables())  # Careful with asyncio.run in module level

    def _initialize_engine(self):
        """Creates the synchronous SQLAlchemy engine."""
        try:
            # connect_args can be used for DuckDB specific settings if needed
            # e.g., connect_args={'read_only': 'false', 'threads': '4'}
            self._engine = create_engine(
                self.database_url, echo=False
            )  # Set echo=True for debugging SQL
            logger.info("SQLAlchemy engine for DuckDB created successfully.")
        except Exception as e:
            logger.exception(
                "Failed to create SQLAlchemy engine for"
                f" {self.database_url}: {e}"
            )
            raise

    # --- Synchronous Helper Methods (Run in Thread Pool) ---

    def _get_sync_session(self) -> Session:
        """Provides a synchronous session context."""
        # If using sessionmaker: return self._session_local()
        return Session(self._engine)

    def _verify_connection_and_tables_sync(self):
        """Synchronously verifies DB connection and checks for essential tables."""
        if not self._engine:
            raise ConnectionError("Database engine not initialized.")
        try:
            with self._get_sync_session() as session:
                # Simple query to test connection
                session.exec(text("SELECT 1"))
                logger.info("Database connection verified.")

                # Check if tables exist using SQLAlchemy Inspector
                inspector = inspect(self._engine)
                required_tables = {
                    DimTime.__tablename__,
                    DimMetric.__tablename__,
                    FactMetricValue.__tablename__,
                }
                existing_tables = set(inspector.get_table_names())
                missing_tables = required_tables - existing_tables

                if missing_tables:
                    logger.warning(
                        f"Missing required tables in database {self.db_path}:"
                        f" {missing_tables}. Run migrations."
                    )
                    # Depending on policy, you could raise an error here
                    # raise RuntimeError(f"Missing required tables: {missing_tables}")
                else:
                    logger.info("Required tables found.")

        except OperationalError as e:
            # Handle specific errors like file not found or permission issues
            logger.error(f"Database operation failed for {self.db_path}: {e}")
            # Check if it's a file not found error specifically
            if "Cannot open file" in str(e):
                logger.error(
                    f"Database file not found at {self.db_path}. Ensure the"
                    " path is correct and the file exists."
                )
            raise ConnectionError(
                f"Failed to connect or verify database {self.db_path}"
            ) from e
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during DB verification: {e}"
            )
            raise ConnectionError(
                "Failed to verify database connection or tables."
            ) from e

    def _ensure_dimensions_sync(
        self,
        session: Session,
        period_key: str,
        metric_key: str,
        # Optional: data to create dimensions if they don't exist
        time_data: Optional[Dict[str, Any]] = None,
        metric_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DimTime, DimMetric]:
        """
        Ensures DimTime and DimMetric exist, creating them if necessary.
        Must be called within an active session transaction.
        Returns the fetched or created dimension objects.
        """
        # --- Handle DimTime ---
        dim_time = session.get(DimTime, period_key)
        if not dim_time:
            if not time_data:
                raise ValueError(
                    f"DimTime with key '{period_key}' not found and no data"
                    " provided to create it."
                )
            try:
                logger.info(
                    f"Attempting to create DimTime for key: {period_key}"
                )
                new_dim_time = DimTime(period_key=period_key, **time_data)
                session.add(new_dim_time)
                session.flush()  # Try to insert it
                dim_time = new_dim_time  # It's now managed and "gotten"
                logger.info(
                    f"DimTime for key '{period_key}' created and flushed."
                )
            except (
                IntegrityError
            ):  # Specific error for duplicate key / constraint violation
                session.rollback()  # Rollback the failed flush attempt for this new object
                logger.warning(
                    f"IntegrityError creating DimTime for '{period_key}',"
                    " likely already exists. Re-fetching."
                )
                dim_time = session.get(
                    DimTime, period_key
                )  # Re-fetch, it MUST exist now
                if (
                    not dim_time
                ):  # Should not happen if IntegrityError was due to duplicate
                    msg = (
                        f"FATAL: DimTime '{period_key}' not found after"
                        " IntegrityError rollback and re-fetch."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)
            except Exception as e:
                session.rollback()
                msg = (
                    f"Unexpected error creating DimTime for {period_key}: {e}"
                )
                logger.exception(msg)
                raise

        # --- Handle DimMetric ---
        dim_metric = session.get(DimMetric, metric_key)
        if not dim_metric:
            if not metric_data:
                raise ValueError(
                    f"DimMetric with key '{metric_key}' not found and no data"
                    " provided to create it."
                )
            try:
                logger.info(
                    f"Attempting to create DimMetric for key: {metric_key}"
                )
                new_dim_metric = DimMetric(
                    metric_key=metric_key, **metric_data
                )
                session.add(new_dim_metric)
                session.flush()
                dim_metric = new_dim_metric
                logger.info(
                    f"DimMetric for key '{metric_key}' created and flushed."
                )
            except IntegrityError:
                session.rollback()
                msg = (
                    f"IntegrityError creating DimMetric for '{metric_key}',"
                    " likely already exists. Re-fetching."
                )
                logger.warning(msg)
                dim_metric = session.get(DimMetric, metric_key)
                if not dim_metric:
                    msg = (
                        f"FATAL: DimMetric '{metric_key}' not found after"
                        " IntegrityError rollback and re-fetch."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)
            except Exception as e:
                session.rollback()
                msg = (
                    "Unexpected error creating DimMetric for"
                    f" {metric_key}: {e}"
                )
                logger.exception(msg)
                raise

        return dim_time, dim_metric

    def _upsert_metric_value_sync(
        self,
        period_key: str,
        metric_key: str,
        numeric_val: Optional[float] = None,
        json_val: Optional[Dict[str, Any]] = None,
        time_data: Optional[
            Dict[str, Any]
        ] = None,  # e.g., {'start_date': ..., 'end_date': ..., 'label': ...}
        metric_data: Optional[
            Dict[str, Any]
        ] = None,  # e.g., {'description': ..., 'originating_report_name': ...}
    ) -> bool:
        """
        Synchronously performs the UPSERT operation for a FactMetricValue.
        Ensures related dimensions exist.
        """
        if not self._engine:
            raise ConnectionError("Database engine not initialized.")

        current_ts = datetime.datetime.now(
            datetime.timezone.utc
        )  # Use timezone-aware UTC

        # SQL for UPSERT using ON CONFLICT (DuckDB/Postgres style)
        # Use native JSON support if possible
        sql_upsert = text("""
            INSERT INTO fact_metric_value (period_key, metric_key, numeric_val, json_val, calc_ts)
            VALUES (:period_key, :metric_key, :numeric_val, :json_val, :calc_ts)
            ON CONFLICT (period_key, metric_key)
            DO UPDATE SET
                numeric_val = EXCLUDED.numeric_val,
                json_val = EXCLUDED.json_val,
                calc_ts = EXCLUDED.calc_ts;
        """)

        try:
            with self._get_sync_session() as session:
                try:
                    # 1. Ensure Dimensions Exist (within the transaction)
                    self._ensure_dimensions_sync(
                        session, period_key, metric_key, time_data, metric_data
                    )

                    # 2. Perform UPSERT using raw SQL for atomicity
                    session.exec(
                        sql_upsert.params(
                            period_key=period_key,
                            metric_key=metric_key,
                            numeric_val=numeric_val,
                            json_val=json_val,
                            calc_ts=current_ts,
                        )
                    )
                    session.commit()
                    logger.debug(
                        f"Upsert successful for {metric_key} in {period_key}"
                    )
                    return True
                except Exception as inner_exc:
                    logger.exception(
                        f"Error during upsert transaction for {metric_key} in"
                        f" {period_key}. Rolling back."
                    )
                    session.rollback()
                    raise inner_exc  # Re-raise the inner exception
        except SQLAlchemyError as e:
            logger.exception(
                f"Database error during upsert for {metric_key} in"
                f" {period_key}: {e}"
            )
            return False
        except ValueError as e:  # Catch errors from _ensure_dimensions_sync
            logger.error(f"Data validation error during upsert: {e}")
            return False

    def _get_metric_value_sync(
        self, period_key: str, metric_key: str
    ) -> Optional[FactMetricValue]:
        """Synchronously fetches a single metric value."""
        if not self._engine:
            raise ConnectionError("Database engine not initialized.")
        try:
            with self._get_sync_session() as session:
                statement = select(FactMetricValue).where(
                    FactMetricValue.period_key == period_key,
                    FactMetricValue.metric_key == metric_key,
                )
                result = session.exec(statement).one_or_none()
                return result
        except SQLAlchemyError as e:
            logger.exception(
                f"Database error fetching metric {metric_key} for"
                f" {period_key}: {e}"
            )
            return None  # Or re-raise depending on desired error handling

    def _get_metrics_for_period_sync(
        self, period_key: str
    ) -> List[FactMetricValue]:
        """Synchronously fetches all metric values for a given period."""
        if not self._engine:
            raise ConnectionError("Database engine not initialized.")
        try:
            with self._get_sync_session() as session:
                statement = select(FactMetricValue).where(
                    FactMetricValue.period_key == period_key
                )
                results = session.exec(statement).all()
                return list(results)  # Convert Sequence to List
        except SQLAlchemyError as e:
            logger.exception(
                "Database error fetching metrics for period"
                f" {period_key}: {e}",
                exc_info=True,
            )
            return []

    def _get_latest_period_key_sync(self, session: Session) -> Optional[str]:
        """Synchronously fetches the most recent period_key from DimTime."""
        if not DimTime:  # Models not loaded
            logger.error(
                "DimTime model not available in _get_latest_period_key_sync."
            )
            return None
        try:
            # Assuming DimTime has an 'end_date' field to determine recency
            # and 'period_key' is the identifier we want.
            # Adjust field names if your DimTime model is different.
            statement = select(DimTime.period_key).order_by(DimTime.end_date.desc()).limit(1)  # type: ignore
            result = session.exec(statement).first()
            return result if result else None
        except SQLAlchemyError as e:
            logger.error(
                f"Database error fetching latest period key: {e}",
                exc_info=True,
            )
            return None
        except (
            AttributeError
        ):  # Catch if DimTime or its attributes are missing
            logger.error(
                "AttributeError in _get_latest_period_key_sync. Ensure DimTime"
                " is imported and has 'period_key' and 'end_date' attributes.",
                exc_info=True,
            )
            return None

    # --- Public Asynchronous Methods ---

    async def verify_connection_and_tables(self) -> None:
        """Asynchronously verifies DB connection and checks for essential tables."""
        logger.info(
            "Verifying database connection and table structure"
            " asynchronously..."
        )
        try:
            await asyncio.to_thread(self._verify_connection_and_tables_sync)
            logger.info("Database verification complete.")
        except Exception as e:
            logger.error(f"Asynchronous database verification failed: {e}")
            # Decide if this should prevent the app from starting
            raise

    async def upsert_metric_value(
        self,
        period_key: str,
        metric_key: str,
        numeric_val: Optional[float] = None,
        json_val: Optional[Dict[str, Any]] = None,
        time_data: Optional[Dict[str, Any]] = None,
        metric_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Asynchronously inserts or updates a metric value for a specific period.
        Ensures related dimensions (DimTime, DimMetric) exist, creating them
        if necessary using the provided optional data.

        Args:
            period_key: The key for the time period (e.g., '2023Q4').
            metric_key: The key for the metric (e.g., 'total_cves').
            numeric_val: The numeric value of the metric (optional).
            json_val: The JSON value of the metric (optional, e.g., a list or dict).
            time_data: Dictionary with data for DimTime if it needs creation
                       (e.g., {'start_date': date, 'end_date': date, 'label': str}).
            metric_data: Dictionary with data for DimMetric if it needs creation
                         (e.g., {'description': str, 'originating_report_name': str}).

        Returns:
            True if the upsert was successful, False otherwise.
        """
        # if numeric_val is None and json_val is None:
        #     logger.warning(f"Upsert attempt for {metric_key} in {period_key} with no value provided.")
        #     # Decide if this should be an error or just return False
        #     return False

        return await asyncio.to_thread(
            self._upsert_metric_value_sync,
            period_key,
            metric_key,
            numeric_val,
            json_val,
            time_data,
            metric_data,
        )

    async def get_metric_value(
        self, period_key: str, metric_key: str
    ) -> Optional[FactMetricValue]:
        """
        Asynchronously retrieves a specific metric value for a given period and key.

        Args:
            period_key: The key for the time period.
            metric_key: The key for the metric.

        Returns:
            A FactMetricValue object if found, otherwise None.
        """
        return await asyncio.to_thread(
            self._get_metric_value_sync, period_key, metric_key
        )

    async def get_metrics_for_period(
        self, period_key: str
    ) -> List[FactMetricValue]:
        """
        Asynchronously retrieves all metric values for a given period.

        Args:
            period_key: The key for the time period.

        Returns:
            A list of FactMetricValue objects for the period. Returns an empty list
            if the period is not found or on database error.
        """
        return await asyncio.to_thread(
            self._get_metrics_for_period_sync, period_key
        )

    async def get_latest_period_key(self) -> Optional[str]:
        """
        Asynchronously retrieves the most recent period_key from the DimTime table.

        Returns:
            The latest period_key string if found, otherwise None.
        """
        if not self._engine:
            logger.error(
                "Database engine not initialized. Cannot get latest period"
                " key."
            )
            return None
        try:
            loop = asyncio.get_running_loop()
            # Use a lambda to properly pass the session from _get_sync_session
            period_key = await loop.run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                lambda: self._get_latest_period_key_sync(
                    self._get_sync_session()
                ),
            )
            return period_key
        except Exception as e:
            logger.error(f"Error in get_latest_period_key: {e}", exc_info=True)
            return None

    async def close(self) -> None:
        """Cleanly disposes of the engine connection pool (if applicable)."""
        if self._engine:
            # For synchronous engines, dispose might not do much for file dbs,
            # but it's good practice if pooling were involved.
            # self._engine.dispose() # Use dispose for pooled connections
            logger.info(
                "DuckDB Metrics Service engine resources hypothetically"
                " released."
            )
            # For DuckDB file, there isn't really a pool to dispose in the same way
            # as server-based DBs. Ensuring sessions are closed is key.
            self._engine = None


# --- Example Usage (Conceptual - within your FastAPI app) ---

# async def save_report_metrics(report_data: Dict, metrics_service: DuckDBMetricsService):
#     period = report_data['period_key'] # e.g., '2024Q1'
#     time_details = report_data['time_details'] # e.g., {'start_date': ..., 'label': ...}
#     report_name = report_data['report_name'] # e.g., 'quarterly_cve_deep_dive'

#     success_count = 0
#     fail_count = 0

#     for metric_key, result in report_data['metrics'].items():
#         metric_details = result['definition'] # e.g., {'description': ...}
#         metric_details['originating_report_name'] = report_name # Add report name

#         num_val = result.get('numeric_value')
#         json_val = result.get('json_value')

#         success = await metrics_service.upsert_metric_value(
#             period_key=period,
#             metric_key=metric_key,
#             numeric_val=num_val,
#             json_val=json_val,
#             time_data=time_details,
#             metric_data=metric_details
#         )
#         if success:
#             success_count += 1
#         else:
#             fail_count += 1

#     logger.info(f"Finished saving metrics for {period}. Success: {success_count}, Failed: {fail_count}")


# async def get_some_metric(period: str, key: str, metrics_service: DuckDBMetricsService):
#     value_obj = await metrics_service.get_metric_value(period, key)
#     if value_obj:
#         print(f"Value for {key} in {period}: {value_obj.numeric_val or value_obj.json_val}")
#         return value_obj
#     else:
#         print(f"Metric {key} for {period} not found.")
#         return None

# --- Test Section ---
if __name__ == "__main__":
    from dotenv import load_dotenv

    # Assuming MetricsDatabasePath is in a module like 'core.config_loader'
    # Adjust the import path as per your project structure.
    try:
        from ..core.config_loader import MetricsDatabasePath
    except ImportError:
        # Fallback for direct script execution if relative import fails
        # This might happen if the script is run from a different context
        # where the relative import isn't resolved. You might need to adjust PYTHONPATH
        # or provide a more robust way to locate MetricsDatabasePath.
        logger.warning(
            "Could not perform relative import for MetricsDatabasePath."
            " Attempting direct import path or placeholder."
        )
        # Placeholder if direct import is complex or for basic testing without full .env setup:

        class MetricsDatabasePath:
            db_path: Optional[str] = (
                "./data/metrics.db"  # Example default, adjust
            )

    async def main():
        """
        Test script for DuckDBMetricsService: Fetches and prints the latest set of metrics.
        """
        load_dotenv()
        db_path_config = MetricsDatabasePath()
        service: Optional[DuckDBMetricsService] = None

        if not db_path_config.db_path:
            logger.error(
                "Could not retrieve metrics database path. Ensure .env is"
                " configured or MetricsDatabasePath provides a default."
                " Exiting."
            )
            return

        logger.info(f"Using database path: {db_path_config.db_path}")
        service = DuckDBMetricsService(db_path_config.db_path)

        try:
            logger.info("Verifying connection and tables...")
            await service.verify_connection_and_tables()
            logger.info("Connection and tables verified.")

            logger.info("Fetching the latest period key...")
            latest_period_key = await service.get_latest_period_key()

            if latest_period_key:
                logger.info(f"Latest period key found: {latest_period_key}")
                logger.info(
                    f"Retrieving all metrics for period: {latest_period_key}"
                )

                all_metrics = await service.get_metrics_for_period(
                    latest_period_key
                )

                if all_metrics:
                    logger.info(
                        f"Found {len(all_metrics)} metrics for"
                        f" {latest_period_key}:"
                    )
                    for metric_value_obj in all_metrics:
                        value_display = (
                            metric_value_obj.numeric_val
                            if metric_value_obj.numeric_val is not None
                            else metric_value_obj.json_val
                        )

                        metric_desc_str = ""
                        # Attempt to access related DimMetric for description if relationship exists
                        if (
                            hasattr(metric_value_obj, 'metric')
                            and metric_value_obj.metric
                            and hasattr(metric_value_obj.metric, 'description')
                        ):
                            metric_desc_str = f" (Desc: {metric_value_obj.metric.description})"  # type: ignore
                        elif (
                            hasattr(metric_value_obj, 'dim_metric')
                            and metric_value_obj.dim_metric
                            and hasattr(
                                metric_value_obj.dim_metric, 'description'
                            )
                        ):
                            metric_desc_str = f" (Desc: {metric_value_obj.dim_metric.description})"  # type: ignore

                        logger.info(
                            "  - Metric Key:"
                            f" {metric_value_obj.metric_key}{metric_desc_str},"
                            f" Value: {value_display}"
                        )
                else:
                    logger.info(
                        "No metrics found for the latest period_key:"
                        f" {latest_period_key}"
                    )
            else:
                logger.info(
                    "No period keys found in the DimTime table. Cannot fetch"
                    " latest metrics."
                )

        except OperationalError as oe:
            logger.error(
                "Database operational error (e.g., file not found,"
                f" permissions, malformed DB): {oe}",
                exc_info=True,
            )
            logger.error(
                "Please ensure the database file exists at"
                f" {db_path_config.db_path}, is a valid DuckDB file, and is"
                " accessible."
            )
        except ConnectionError as ce:
            logger.error(f"Database connection error: {ce}", exc_info=True)
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during the metrics service"
                f" test: {e}"
            )
        finally:
            if service:
                logger.info("Closing metrics service connection...")
                await service.close()
                logger.info("Connection closed.")
            logger.info("Metrics service test finished.")

    # Run the async main function
    asyncio.run(main())
