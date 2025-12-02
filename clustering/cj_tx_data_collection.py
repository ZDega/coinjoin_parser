"""
CoinJoin Transaction Data Collection Script

This script collects CoinJoin transaction data from Bitcoin APIs and stores it
in the DuckDB database according to the schema defined in data/create_data_base.py
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import duckdb
from tqdm import tqdm

# Import existing API client
from api_clients.mempool_client import MempoolClient

# Import Pydantic models
from models import (
    ScriptPubkeyType,
    TransactionInput,
    TransactionOutput,
    Coordinator,
    InputAddress,
    OutputAddress,
    CoinjoinTransaction,
    DatabaseConnection,
    TransactionProcessor,
)

load_dotenv()


# ============================================================================
# MAIN COLLECTION WORKFLOW
# ============================================================================
def process_coinjoins_for_scripttype() -> bool:


    with DatabaseConnection() as db_conn:
        coinjoin_transactions = db_conn.conn.execute("""SELECT to_hex(tx_id), coordinator_endpoint
                                                        FROM raw_round_data
                                                        WHERE processed = FALSE;
                                                    """).fetchall()
        mempool_client = MempoolClient()
        total_transactions = len(coinjoin_transactions)

        for tx_id, coordinator_endpoint in tqdm(coinjoin_transactions, desc=f"Processing {total_transactions} unprocessed CoinJoin transactions"):
            tx_id: str
            coordinator_endpoint: str

            raw_transation_data = mempool_client.get_transaction(tx_id)

            coor_id = db_conn.get_or_create_coordinator_id(coordinator_endpoint=coordinator_endpoint)

            cj_tx = TransactionProcessor.process_transaction(tx_raw=raw_transation_data, coordinator_id=coor_id, db_conn=db_conn)

            #TODO add execute statement for cj_tx processed update to TRUE




# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """
    Main entry point for the data collection script.

    Provides CLI interface for collecting CoinJoin transaction data.
    """
    # TODO: Implement CLI with argparse
    # Support modes: single transaction, batch, file
    # Display statistics and handle errors
    pass


if __name__ == "__main__":
    main()
