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

        print(f"\n🔍 Found {total_transactions} unprocessed CoinJoin transactions to process\n")

        for tx_id, coordinator_endpoint in tqdm(coinjoin_transactions, desc="📊 Processing CoinJoin transactions", unit="tx"):
            tx_id: str
            coordinator_endpoint: str

            try:
                # Begin atomic transaction
                db_conn.conn.begin()
                print(f"\n⚡ Starting transaction: {tx_id[:16]}...")

                raw_transation_data = mempool_client.get_transaction(tx_id)
                print(f"  📥 Fetched raw transaction data from mempool")

                coor_id = db_conn.get_or_create_coordinator_id(coordinator_endpoint=coordinator_endpoint)
                print(f"  🎯 Coordinator ID: {coor_id} ({coordinator_endpoint})")

                cj_tx = TransactionProcessor.process_transaction(tx_raw=raw_transation_data, coordinator_id=coor_id, db_conn=db_conn)
                print(f"  ⚙️  Processed {cj_tx.number_inputs} inputs and {cj_tx.number_outputs} outputs")

                # Insert the processed CoinJoin transaction
                db_conn.insert_coinjoin_transaction(cj_tx)
                print(f"  💾 Inserted CoinJoin transaction into database")

                # Mark the transaction as processed in raw_round_data
                db_conn.conn.execute(
                    "UPDATE raw_round_data SET processed = TRUE WHERE tx_id = from_hex(?)",
                    [tx_id]
                )
                print(f"  ✅ Marked transaction as processed in raw_round_data")

                # Commit all changes atomically
                db_conn.conn.commit()
                print(f"  🎉 Successfully committed all changes for tx {tx_id[:16]}...\n")

            except Exception as e:
                # Rollback all changes if any operation fails
                db_conn.conn.rollback()
                print(f"  ❌ Error processing transaction {tx_id}: {e}")
                print(f"  🔄 Rolled back all changes for this transaction\n")
                # Continue processing other transactions
                continue

        print(f"\n✨ Completed processing all transactions!\n")




# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """
    Main entry point for the data collection script.

    Provides CLI interface for collecting CoinJoin transaction data.
    """
    process_coinjoins_for_scripttype()


if __name__ == "__main__":
    main()
