import pytest
import duckdb

from data import init_database_schema
from api_clients import MempoolClient
from models import DatabaseConnection, TransactionProcessor
import tqdm



@pytest.fixture
def db_conn():
    """
    In-memory DuckDB with full schema (enums, structs, tables).
    Fresh for each test.
    """
    conn = duckdb.connect(database=":memory:")

    # Reuse the actual schema creation logic
    init_database_schema(conn)

    try:
        yield conn
    finally:
        conn.close()

#Real test data for coinjoin rounds
@pytest.fixture
def raw_round_data():
    """Test data for coinjoin rounds."""
    return [
        {
            "CoordinatorEndpoint": "https://coinjoin.kruw.io/",
            "EstimatedCoordinatorEarningsSats": 11829,
            "RoundId": "e301f66cc721edab57a9e76c0f8987e96c1bbb1a03dcf3cd51cbb3b2d806fc99",
            "IsBlame": True,
            "CoordinationFeeRate": 0,
            "MinInputCount": 100,
            "ParametersMiningFeeRate": 2,
            "RoundStartTime": "2025-12-02T12:05:30+00:00",
            "RoundEndTime": "2025-12-02T12:10:10+00:00",
            "TxId": "75064de0c90be0594b23b6af53f5e5c9b888e8eb21976881e5ed11ff827ff319",
            "FinalMiningFeeRate": 2.01,
            "VirtualSize": 32810,
            "TotalMiningFee": 65905,
            "InputCount": 301,
            "TotalInputAmount": 4034831072,
            "FreshInputsEstimateBtc": 0.88997804,
            "AverageStandardInputsAnonSet": 7.44,
            "OutputCount": 359,
            "TotalOutputAmount": 4034765167,
            "ChangeOutputsAmountRatio": 0.0342,
            "AverageStandardOutputsAnonSet": 13.54,
            "TotalLeftovers": 11829,
        },
        {
            "CoordinatorEndpoint": "https://coinjoin.kruw.io/",
            "EstimatedCoordinatorEarningsSats": 27811,
            "RoundId": "63c0f071387eb83701aba5b453c3be42f9691d62c5fd2c2bc7b97fe3c02bb501",
            "IsBlame": False,
            "CoordinationFeeRate": 0,
            "MinInputCount": 100,
            "ParametersMiningFeeRate": 1,
            "RoundStartTime": "2025-12-02T12:00:03+00:00",
            "RoundEndTime": "2025-12-02T12:53:42+00:00",
            "TxId": "7df9c364d3628f8f68332d41000596623bfca29e2fe1532a1c18a5d7d77b4a68",
            "FinalMiningFeeRate": 1,
            "VirtualSize": 26822,
            "TotalMiningFee": 26938,
            "InputCount": 233,
            "TotalInputAmount": 7078695374,
            "FreshInputsEstimateBtc": 1.19881405,
            "AverageStandardInputsAnonSet": 6.06,
            "OutputCount": 310,
            "TotalOutputAmount": 7078668436,
            "ChangeOutputsAmountRatio": 0,
            "AverageStandardOutputsAnonSet": 9.97,
            "TotalLeftovers": 27811,
        },
        {
            "CoordinatorEndpoint": "https://coinjoin.kruw.io/",
            "EstimatedCoordinatorEarningsSats": 33057,
            "RoundId": "dd9e0a3fdea625f1c28d1dc80dbea21eab76ca341622d690182e82027e537c23",
            "IsBlame": True,
            "CoordinationFeeRate": 0,
            "MinInputCount": 100,
            "ParametersMiningFeeRate": 3,
            "RoundStartTime": "2025-12-02T13:43:56+00:00",
            "RoundEndTime": "2025-12-02T13:50:02+00:00",
            "TxId": "0c396ec5b81d7fb95e73049e64033ea497ba0f92cc6220350a9cb9a139c803e7",
            "FinalMiningFeeRate": 3.01,
            "VirtualSize": 25454,
            "TotalMiningFee": 76663,
            "InputCount": 232,
            "TotalInputAmount": 2385100400,
            "FreshInputsEstimateBtc": 2.06925712,
            "AverageStandardInputsAnonSet": 6.11,
            "OutputCount": 285,
            "TotalOutputAmount": 2385023737,
            "ChangeOutputsAmountRatio": 0.0102,
            "AverageStandardOutputsAnonSet": 11.2,
            "TotalLeftovers": 33057,
        },
        {
            "CoordinatorEndpoint": "https://coinjoin.kruw.io/",
            "EstimatedCoordinatorEarningsSats": 54918,
            "RoundId": "c9e383ef124c0fa1cf2361405e494dfb2bc6c04aa93fdd3a5d890ac59c305454",
            "IsBlame": True,
            "CoordinationFeeRate": 0,
            "MinInputCount": 100,
            "ParametersMiningFeeRate": 2,
            "RoundStartTime": "2025-12-02T15:00:11+00:00",
            "RoundEndTime": "2025-12-02T15:06:02+00:00",
            "TxId": "2da13e0301da0d0f0f2365e363a28a15e700380d5ebeb7044ea5382aa1401766",
            "FinalMiningFeeRate": 2.01,
            "VirtualSize": 27167,
            "TotalMiningFee": 54563,
            "InputCount": 230,
            "TotalInputAmount": 2599145664,
            "FreshInputsEstimateBtc": 8.34251397,
            "AverageStandardInputsAnonSet": 5.94,
            "OutputCount": 324,
            "TotalOutputAmount": 2599091101,
            "ChangeOutputsAmountRatio": 0.0069,
            "AverageStandardOutputsAnonSet": 10.53,
            "TotalLeftovers": 54918,
        },
        {
            "CoordinatorEndpoint": "https://coinjoin.kruw.io/",
            "EstimatedCoordinatorEarningsSats": 17580,
            "RoundId": "14d4d625045c4ef94ccfcb79284278fe55b1fda2b9b1402d845a0ea165d9e75d",
            "IsBlame": True,
            "CoordinationFeeRate": 0,
            "MinInputCount": 100,
            "ParametersMiningFeeRate": 1,
            "RoundStartTime": "2025-12-02T15:49:08+00:00",
            "RoundEndTime": "2025-12-02T15:53:56+00:00",
            "TxId": "0d2696cf6fe755b5763f189d80589f65f839e30dbc6d361145a026ec1576e334",
            "FinalMiningFeeRate": 1,
            "VirtualSize": 28054,
            "TotalMiningFee": 28192,
            "InputCount": 263,
            "TotalInputAmount": 8204325656,
            "FreshInputsEstimateBtc": 6.80190073,
            "AverageStandardInputsAnonSet": 6.68,
            "OutputCount": 296,
            "TotalOutputAmount": 8204297464,
            "ChangeOutputsAmountRatio": 0.0187,
            "AverageStandardOutputsAnonSet": 10.36,
            "TotalLeftovers": 17580,
        },
    ]
#fake test data for coinjoin rounds
@pytest.fixture
def raw_fake_data():
    """Test data for coinjoin rounds."""
    return [
        {
            "CoordinatorEndpoint": "https://coinjoin.kruw.io/",
            "EstimatedCoordinatorEarningsSats": 11829,
            "RoundId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "IsBlame": True,
            "CoordinationFeeRate": 0,
            "MinInputCount": 2,
            "ParametersMiningFeeRate": 2,
            "RoundStartTime": "2025-12-02T12:05:30+00:00",
            "RoundEndTime": "2025-12-02T12:10:10+00:00",
            "TxId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "FinalMiningFeeRate": 2.01,
            "VirtualSize": 20,
            "TotalMiningFee": 1200,
            "InputCount": 4,
            "TotalInputAmount": 4034831072,
            "FreshInputsEstimateBtc": 0.88997804,
            "AverageStandardInputsAnonSet": 7.44,
            "OutputCount": 5,
            "TotalOutputAmount": 4034765167,
            "ChangeOutputsAmountRatio": 0.0342,
            "AverageStandardOutputsAnonSet": 13.54,
            "TotalLeftovers": 11829,
        },
        {
            "CoordinatorEndpoint": "https://coinjoin.kruw.io/",
            "EstimatedCoordinatorEarningsSats": 27811,
            "RoundId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "IsBlame": False,
            "CoordinationFeeRate": 0,
            "MinInputCount": 2,
            "ParametersMiningFeeRate": 1,
            "RoundStartTime": "2025-12-02T12:00:03+00:00",
            "RoundEndTime": "2025-12-02T12:53:42+00:00",
            "TxId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "FinalMiningFeeRate": 1,
            "VirtualSize": 26822,
            "TotalMiningFee": 26938,
            "InputCount": 3,
            "TotalInputAmount": 7078695374,
            "FreshInputsEstimateBtc": 1.19881405,
            "AverageStandardInputsAnonSet": 6.06,
            "OutputCount": 4,
            "TotalOutputAmount": 7078668436,
            "ChangeOutputsAmountRatio": 0,
            "AverageStandardOutputsAnonSet": 9.97,
            "TotalLeftovers": 27811,
        },
    ]

@pytest.fixture
def db_with_data(db_conn, raw_round_data):
    """Tables filled with minimal test data."""
    conn = db_conn
    conn: duckdb.DuckDBPyConnection

    # Insert minimal data into raw_round_data table
    insert_sql = """
        INSERT OR IGNORE INTO raw_round_data (
            coordinator_endpoint, estimated_coordinator_earnings_sats, round_id, isBlame,
            coordinaton_fee_rate, min_input_count, parameters_mining_fee_rate,
            round_start_time, round_end_time, tx_id, final_mining_fee_rate, virtual_size,
            total_mining_fee, input_count, total_input_amount, fresh_inputs_estimate_btc,
            average_standard_input_anon_set, output_count, total_output_amount,
            change_output_ratio, average_standard_output_anon_set, total_left_overs
        )
        VALUES (
            ?, ?, from_hex(?), ?, ?, ?, ?, ?, ?, from_hex(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
    """

    insert_params = [
        (
            row["CoordinatorEndpoint"],
            row["EstimatedCoordinatorEarningsSats"],
            row["RoundId"],
            row["IsBlame"],
            row["CoordinationFeeRate"],
            row["MinInputCount"],
            row["ParametersMiningFeeRate"],
            row["RoundStartTime"],
            row["RoundEndTime"],
            row["TxId"],
            row["FinalMiningFeeRate"],
            row["VirtualSize"],
            row["TotalMiningFee"],
            row["InputCount"],
            row["TotalInputAmount"],
            row["FreshInputsEstimateBtc"],
            row["AverageStandardInputsAnonSet"],
            row["OutputCount"],
            row["TotalOutputAmount"],
            row["ChangeOutputsAmountRatio"],
            row["AverageStandardOutputsAnonSet"],
            row["TotalLeftovers"],
        )
        for row in raw_round_data
    ]

    conn.executemany(insert_sql, insert_params)

    return conn

@pytest.fixture
def db_with_fake_data(db_conn, raw_fake_data):
    """Tables filled with fake test data."""
    conn = db_conn
    conn: duckdb.DuckDBPyConnection

    # Insert minimal data into raw_round_data table
    insert_sql = """
        INSERT OR IGNORE INTO raw_round_data (
            coordinator_endpoint, estimated_coordinator_earnings_sats, round_id, isBlame,
            coordinaton_fee_rate, min_input_count, parameters_mining_fee_rate,
            round_start_time, round_end_time, tx_id, final_mining_fee_rate, virtual_size,
            total_mining_fee, input_count, total_input_amount, fresh_inputs_estimate_btc,
            average_standard_input_anon_set, output_count, total_output_amount,
            change_output_ratio, average_standard_output_anon_set, total_left_overs
        )
        VALUES (
            ?, ?, from_hex(?), ?, ?, ?, ?, ?, ?, from_hex(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
    """

    insert_params = [
        (
            row["CoordinatorEndpoint"],
            row["EstimatedCoordinatorEarningsSats"],
            row["RoundId"],
            row["IsBlame"],
            row["CoordinationFeeRate"],
            row["MinInputCount"],
            row["ParametersMiningFeeRate"],
            row["RoundStartTime"],
            row["RoundEndTime"],
            row["TxId"],
            row["FinalMiningFeeRate"],
            row["VirtualSize"],
            row["TotalMiningFee"],
            row["InputCount"],
            row["TotalInputAmount"],
            row["FreshInputsEstimateBtc"],
            row["AverageStandardInputsAnonSet"],
            row["OutputCount"],
            row["TotalOutputAmount"],
            row["ChangeOutputsAmountRatio"],
            row["AverageStandardOutputsAnonSet"],
            row["TotalLeftovers"],
        )
        for row in raw_fake_data
    ]

    conn.executemany(insert_sql, insert_params)

    return conn

@pytest.fixture
def mempool_client():
    """MempoolClient instance for tests."""
    client = MempoolClient()

    try:
        yield client
    finally:
        client.close()

def test_testdata_fixture(db_with_data):
    """Sanity check that test data fixture works."""
    conn = db_with_data
    conn: duckdb.DuckDBPyConnection

    row = conn.execute(
        "SELECT COUNT(*) FROM raw_round_data;"
    ).fetchone()

    assert row[0] == 5  # We inserted 5 rows

def test_testdata_coordinator(db_with_data):
    """Check that coordinator_endpoint values are correct."""
    conn = db_with_data
    conn: duckdb.DuckDBPyConnection

    rows = conn.execute(
        "SELECT DISTINCT coordinator_endpoint FROM raw_round_data;"
    ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == "https://coinjoin.kruw.io/"



def test_script_pubkey_type_enum_exists(db_conn):
    row = db_conn.execute(
        "SELECT enum_range(NULL::script_pubkey_type);"
    ).fetchone()

    # row[0] is something like ['p2pkh', 'p2sh', ...]
    assert "p2pkh" in row[0]
    assert "p2sh" in row[0]
    assert "v0_p2wpkh" in row[0]
    assert "v0_p2wsh" in row[0]
    assert "v1_p2tr" in row[0]
    assert "other" in row[0]


def test_mempool_client_get_transaction(mempool_client):
    """Test MempoolClient.get_transaction method."""
    client = mempool_client
    client: MempoolClient

    # Example txid (a known Bitcoin transaction)
    txid = "75064de0c90be0594b23b6af53f5e5c9b888e8eb21976881e5ed11ff827ff319"

    # Call the method
    try:
        tx_data = client.get_transaction(txid)
    except Exception as e:
        pytest.fail(f"get_transaction raised an exception: {e}")

    # Basic assertions about the returned data
    assert isinstance(tx_data, dict)
    assert tx_data.get("txid") == txid
    assert "vin" in tx_data
    assert "vout" in tx_data




def test_coinjoin_tx_processor_with_real_data(mempool_client, db_with_data):
    """Test CoinJoin transaction processing logic."""
    client = mempool_client
    client: MempoolClient
    conn = db_with_data
    conn: duckdb.DuckDBPyConnection

    # Here you would call your CoinJoin processing logic
    # For example:
    # processed_data = process_coinjoin_transaction(tx_data)
    with DatabaseConnection(connection=conn) as db_conn:
        coinjoin_transactions = db_conn.conn.execute("""SELECT to_hex(tx_id), coordinator_endpoint
                                                        FROM raw_round_data
                                                        WHERE processed = FALSE;
                                                    """).fetchall()
        mempool_client = MempoolClient()
        total_transactions = len(coinjoin_transactions)
        

        print(f"\n🔍 Found {total_transactions} unprocessed CoinJoin transactions to process\n")
        assert total_transactions == 5, "Expected 5 unprocessed CoinJoin transactions to process"

        for tx_id, coordinator_endpoint in tqdm.tqdm(coinjoin_transactions, desc="📊 Processing CoinJoin transactions", unit="tx"):
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
        
        # Print coinjoin transactions table
        print("\n📊 Coinjoin Transactions Table:")
        print("=" * 120)
        cj_txs = db_conn.conn.execute("""
            SELECT tx_id_int,
                   LOWER(to_hex(tx_id)) as tx_id_hex,
                   number_inputs,
                   number_outputs,
                   value_inputs,
                   value_outputs,
                   transaction_fee,
                   block_number,
                   processed
            FROM coinjoin_transactions
        """).fetchall()

        print(f"{'ID':<6} {'TX_ID':<66} {'Inputs':<8} {'Outputs':<8} {'In Value':<12} {'Out Value':<12} {'Fee':<10} {'Block':<8} {'Processed'}")
        print("-" * 120)
        for row in cj_txs:
            tx_id_int, tx_id_hex, n_in, n_out, val_in, val_out, fee, block, proc = row
            print(f"{tx_id_int:<6} {tx_id_hex:<66} {n_in:<8} {n_out:<8} {val_in:<12} {val_out:<12} {fee:<10} {block:<8} {proc}")
        print("=" * 120)
        print(f"📝 Note: Input/output lists represented as [...] for brevity\n")

        # Run assertions
        print("🔍 Running assertions...")

        all_processed = db_conn.conn.execute("""SELECT COUNT(*)
                                                        FROM raw_round_data
                                                        WHERE processed = FALSE;
                                                    """).fetchall()

        print("  ✓ Checking all transactions are processed...")
        assert all_processed[0][0] == 0, f"All CoinJoin transactions should be marked as processed, but found {all_processed[0]} unprocessed."
        print("    ✅ All transactions marked as processed")

        processed_cj_txs = db_conn.conn.execute("""SELECT COUNT(*)
                                                        FROM coinjoin_transactions;
                                                    """).fetchone()
        print(f"  ✓ Checking coinjoin transaction count...")
        assert processed_cj_txs[0] == 5, f"Expected 5 processed CoinJoin transactions, found {processed_cj_txs[0]}"
        print(f"    ✅ Found {processed_cj_txs[0]} CoinJoin transactions")

        input_count = db_conn.conn.execute("""SELECT COUNT(*)
                                                FROM input_addresses;
                                            """).fetchone()
        print(f"  ✓ Checking input count...")
        assert input_count[0] < 1312, "Expected fewer than 1312 inputs across all CoinJoin transactions"
        print(f"    ✅ Found {input_count[0]} inputs (< 1312)")

        output_count = db_conn.conn.execute("""SELECT COUNT(*)
                                                FROM output_addresses;
                                            """).fetchone()
        print(f"  ✓ Checking output count...")
        assert output_count[0] < 1575, "Expected fewer than 1574 outputs across all CoinJoin transactions"
        print(f"    ✅ Found {output_count[0]} outputs (< 1574)")

        coordinator_count = db_conn.conn.execute("""SELECT COUNT(*)
                                                 FROM coordinators;
                                            """).fetchone()
        print(f"  ✓ Checking coordinator count...")
        assert coordinator_count[0] == 1, f"Expected 1 coordinator, found {coordinator_count[0]}"
        print(f"    ✅ Found {coordinator_count[0]} coordinator")

        print("\n✅ All assertions passed!")


        print(f"\n✨ Completed processing all transactions!\n")


def mock_api_get_transaction(tx_id: str)-> dict:
    """
    Mock function to return fake transaction data based on transaction ID.

    Args:
        tx_id: Transaction ID (hex string)

    Returns:
        Fake transaction data dict if tx_id is all a's or all b's, None otherwise
    """
    # Transaction ID with all a's
    tx_id = tx_id.lower()
    if tx_id == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa":
        return {
            "txid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "version": 2,
            "locktime": 0,
            "vin": [
                {
                    "txid": "1111111111111111111111111111111111111111111111111111111111111111",
                    "vout": 0,
                    "prevout": {
                        "scriptpubkey": "0014abcd...",
                        "scriptpubkey_address": "bc1qtest1address1111111111111111111111111111",
                        "scriptpubkey_type": "v0_p2wpkh",
                        "value": 101
                    },
                    "is_coinbase": False
                },
                {
                    "txid": "2222222222222222222222222222222222222222222222222222222222222222",
                    "vout": 1,
                    "prevout": {
                        "scriptpubkey": "0014efgh...",
                        "scriptpubkey_address": "bc1qtest2address2222222222222222222222222222",
                        "scriptpubkey_type": "v0_p2wpkh",
                        "value": 101
                    },
                    "is_coinbase": False
                },
                {
                    "txid": "1111111111111111111111111111111111111111111111111111111111111111",
                    "vout": 5,
                    "prevout": {
                        "scriptpubkey": "0014abcd...",
                        "scriptpubkey_address": "bc1qtest3address1111111111111111111111111111",
                        "scriptpubkey_type": "v0_p2wpkh",
                        "value": 101
                    },
                    "is_coinbase": False
                },
                {
                    "txid": "1111111111111111111111111111111111111111111111111111111111111111",
                    "vout": 7,
                    "prevout": {
                        "scriptpubkey": "0014abcd...",
                        "scriptpubkey_address": "bc1qtest4address1111111111111111111111111111",
                        "scriptpubkey_type": "v0_p2wpkh",
                        "value": 101
                    },
                    "is_coinbase": False
                }
            ],
            "vout": [
                {
                    "scriptpubkey": "0014ijkl...",
                    "scriptpubkey_address": "bc1qtest5address3333333333333333333333333333",
                    "scriptpubkey_type": "v0_p2wpkh",
                    "value": 100
                },
                {
                    "scriptpubkey": "0014mnop...",
                    "scriptpubkey_address": "bc1qtest6address4444444444444444444444444444",
                    "scriptpubkey_type": "v0_p2wpkh",
                    "value": 100
                }
                ,
                {
                    "scriptpubkey": "0014mnop...",
                    "scriptpubkey_address": "bc1qtest7address4444444444444444444444444444",
                    "scriptpubkey_type": "v0_p2wpkh",
                    "value": 50
                }
                ,
                {
                    "scriptpubkey": "0014mnop...",
                    "scriptpubkey_address": "bc1qtest8address4444444444444444444444444444",
                    "scriptpubkey_type": "v0_p2wpkh",
                    "value": 50
                }
                ,
                {
                    "scriptpubkey": "0014mnop...",
                    "scriptpubkey_address": "bc1qtest9address4444444444444444444444444444",
                    "scriptpubkey_type": "v0_p2wpkh",
                    "value": 100
                }
            ],
            "size": 1500,
            "weight": 6000,
            "fee": 4,
            "status": {
                "confirmed": True,
                "block_height": 800000,
                "block_hash": "0000000000000000000a7b3c4d5e6f7890abcdef1234567890abcdef1234567",
                "block_time": 1701619530
            }            
        }

    # Transaction ID with all b's
    elif tx_id == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb":
        return {
            "txid": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "version": 2,
            "locktime": 0,
            "vin": [
                {
                    "txid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "vout": 4,
                    "prevout": {
                        "scriptpubkey": "50014mnop...",
                        "scriptpubkey_address": "bc1qtest9address4444444444444444444444444444",
                        "scriptpubkey_type": "v0_p2wpkh",
                        "value": 102
                    },
                    "is_coinbase": False
                },
                {
                    "txid": "1111111111111111111111111111111111111111111111111111111111111111",
                    "vout": 1,
                    "prevout": {
                        "scriptpubkey": "0014abcd...",
                        "scriptpubkey_address": "bc1qtest1address1111111111111111111111111111",
                        "scriptpubkey_type": "v0_p2wpkh",
                        "value": 102
                    },
                    "is_coinbase": False
                },
                {
                    "txid": "1111111111111111111111111111111111111111111111111111111111111111",
                    "vout": 8,
                    "prevout": {
                        "scriptpubkey": "0014abcd...",
                        "scriptpubkey_address": "bc1qtest10address1111111111111111111111111111",
                        "scriptpubkey_type": "v0_p2wpkh",
                        "value": 102
                    },
                    "is_coinbase": False
                },
            ],
            "vout": [
                {
                    "scriptpubkey": "0014abcd...",
                    "scriptpubkey_address": "bc1qtest3address1111111111111111111111111111",
                    "scriptpubkey_type": "v0_p2wpkh",
                    "value": 101
                },
                {
                    "scriptpubkey": "0014cdef...",
                    "scriptpubkey_address": "bc1qtest11address8888888888888888888888888888",
                    "scriptpubkey_type": "v0_p2wpkh",
                    "value": 50
                },
                {
                    "scriptpubkey": "0014cdef...",
                    "scriptpubkey_address": "bc1qtest12address8888888888888888888888888888",
                    "scriptpubkey_type": "v0_p2wpkh",
                    "value": 50
                },
                {
                    "scriptpubkey": "0014cdef...",
                    "scriptpubkey_address": "bc1qtest13address8888888888888888888888888888",
                    "scriptpubkey_type": "v0_p2wpkh",
                    "value": 100
                }
            ],
            "size": 2000,
            "weight": 8000,
            "fee": 5,
            "status": {
                "confirmed": True,
                "block_height": 800001,
                "block_hash": "0000000000000000000a7b3c4d5e6f7892abcdef1234567890abcdef1234567",
                "block_time": 1701619600
            }            
        }

    # Return None for unknown transaction IDs
    return None


def test_coinjoin_tx_processor_with_fake_data(mempool_client, db_with_fake_data):
    """Test CoinJoin transaction processing logic with fake data."""
    client = mempool_client
    client: MempoolClient
    conn = db_with_fake_data
    conn: duckdb.DuckDBPyConnection

    # Here you would call your CoinJoin processing logic
    # For example:
    # processed_data = process_coinjoin_transaction(tx_data)
    with DatabaseConnection(connection=conn) as db_conn:
        coinjoin_transactions = db_conn.conn.execute("""SELECT to_hex(tx_id), coordinator_endpoint
                                                        FROM raw_round_data
                                                        WHERE processed = FALSE;
                                                    """).fetchall()
        
        total_transactions = len(coinjoin_transactions)
        

        print(f"\n🔍 Found {total_transactions} unprocessed CoinJoin transactions to process\n")
        assert total_transactions == 2, f"Expected 2 unprocessed CoinJoin transactions to process, but found {total_transactions}"

        for tx_id, coordinator_endpoint in tqdm.tqdm(coinjoin_transactions, desc="📊 Processing CoinJoin transactions", unit="tx"):
            tx_id: str
            coordinator_endpoint: str

            try:
                # Begin atomic transaction
                db_conn.conn.begin()
                print(f"\n⚡ Starting transaction: {tx_id[:16]}...")

                raw_transation_data = mock_api_get_transaction(tx_id)
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
        
        # Print coinjoin transactions table
        print("\n📊 Coinjoin Transactions Table:")
        print("=" * 120)
        cj_txs = db_conn.conn.execute("""
            SELECT tx_id_int,
                   LOWER(to_hex(tx_id)) as tx_id_hex,
                   number_inputs,
                   number_outputs,
                   value_inputs,
                   value_outputs,
                   transaction_fee,
                   block_number,
                   processed
            FROM coinjoin_transactions
        """).fetchall()

        print(f"{'ID':<6} {'TX_ID':<66} {'Inputs':<8} {'Outputs':<8} {'In Value':<12} {'Out Value':<12} {'Fee':<10} {'Block':<8} {'Processed'}")
        print("-" * 120)
        for row in cj_txs:
            tx_id_int, tx_id_hex, n_in, n_out, val_in, val_out, fee, block, proc = row
            print(f"{tx_id_int:<6} {tx_id_hex:<66} {n_in:<8} {n_out:<8} {val_in:<12} {val_out:<12} {fee:<10} {block:<8} {proc}")
        print("=" * 120)
        print(f"📝 Note: Input/output lists represented as [...] for brevity\n")
        

        #Print input addresses table
        print("\n📊 Input Addresses Table:")
        input_addresses = db_conn.conn.execute("""SELECT input_address_id, address, used_as_output, script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj
                                                FROM input_addresses;
                                            """).fetchall()
        print(f"{'ID':<6} {'ADDRESS':<60} {'USED AS OUTPUT':<15} {'SCRIPT TYPE':<12} {'# of times USED as INPUT':<12} {'TOTAL AMOUNT'} ")
        print("-" * 120)
        for row in input_addresses:
            input_addr_id, input_addr, used_output, script_type, number_of_cj_tx_used_in, total_amount_spent = row
            print(f"{input_addr_id:<6} {input_addr:<60} {used_output:<15} {script_type:<12} {number_of_cj_tx_used_in:<12} {total_amount_spent}")
        print("=" * 120)


        #Print output addresses table
        print("\n📊 Output Addresses Table:")
        output_addresses = db_conn.conn.execute("""SELECT output_address_id, address, used_as_input, script_type, number_of_cjs_used_in_as_output, total_amount_received_in_cj
                                                FROM output_addresses;
                                            """).fetchall()
        print(f"{'ID':<6} {'ADDRESS':<60} {'USED AS INPUT':<15} {'SCRIPT TYPE':<12} {'# of times USED as OUTPUT':<12} {'TOTAL AMOUNT'} ")
        print("-" * 120)
        for row in output_addresses:
            output_addr_id, output_addr, used_input, script_type, number_of_cj_tx_used_in, total_amount_received = row
            print(f"{output_addr_id:<6} {output_addr:<60} {used_input:<15} {script_type:<12} {number_of_cj_tx_used_in:<12} {total_amount_received}")
        print("=" * 120)







        # Run assertions
        print("🔍 Running assertions...")

        all_processed = db_conn.conn.execute("""SELECT COUNT(*)
                                                        FROM raw_round_data
                                                        WHERE processed = FALSE;
                                                    """).fetchall()

        print("  ✓ Checking all transactions are processed...")
        assert all_processed[0][0] == 0, f"All CoinJoin transactions should be marked as processed, but found {all_processed[0]} unprocessed."
        print("    ✅ All transactions marked as processed")

        processed_cj_txs = db_conn.conn.execute("""SELECT COUNT(*)
                                                        FROM coinjoin_transactions;
                                                    """).fetchone()
        print(f"  ✓ Checking coinjoin transaction count...")
        assert processed_cj_txs[0] == 2, f"Expected 2 processed CoinJoin transactions, found {processed_cj_txs[0]}"
        print(f"    ✅ Found {processed_cj_txs[0]} CoinJoin transactions")

        input_count = db_conn.conn.execute("""SELECT COUNT(*)
                                                FROM input_addresses;
                                            """).fetchone()
        print(f"  ✓ Checking input count...")
        assert input_count[0] == 6, "Expected 6 inputs across all CoinJoin transactions"
        print(f"    ✅ Found {input_count[0]} inputs")

        output_count = db_conn.conn.execute("""SELECT COUNT(*)
                                                FROM output_addresses;
                                            """).fetchone()
        print(f"  ✓ Checking output count...")
        assert output_count[0] == 9, "Expected 9 outputs across all CoinJoin transactions"
        print(f"    ✅ Found {output_count[0]} outputs")

        coordinator_count = db_conn.conn.execute("""SELECT COUNT(*)
                                                 FROM coordinators;
                                            """).fetchone()
        print(f"  ✓ Checking coordinator count...")
        assert coordinator_count[0] == 1, f"Expected 1 coordinator, found {coordinator_count[0]}"
        print(f"    ✅ Found {coordinator_count[0]} coordinator")


        both_input_output_addresses_inputs = db_conn.conn.execute("""SELECT COUNT(*)
                                                                    FROM input_addresses
                                                                    WHERE used_as_output = TRUE;
                                                                """).fetchone()
        
        print(f"  ✓ Checking input addresses also used as outputs...")
        assert both_input_output_addresses_inputs[0] == 2, f"Expected 2 input addresses also used as outputs, found {both_input_output_addresses_inputs[0]}"
        print(f"    ✅ Found {both_input_output_addresses_inputs[0]} input addresses also used as outputs")

        both_input_output_addresses_outputs = db_conn.conn.execute("""SELECT COUNT(*)
                                                                    FROM output_addresses
                                                                    WHERE used_as_input = TRUE;
                                                                """).fetchone()
        
        print(f"  ✓ Checking output addresses also used as inputs...")
        assert both_input_output_addresses_outputs[0] == 2, f"Expected 2 output addresses also used as inputs, found {both_input_output_addresses_outputs[0]}"
        print(f"    ✅ Found {both_input_output_addresses_outputs[0]} output addresses also used as inputs")


        input_used_twice = db_conn.conn.execute("""SELECT input_address_id
                                                FROM input_addresses
                                                WHERE number_of_cjs_used_in_as_input > 1;
                                            """).fetchall()
        
        print(f"  ✓ Checking input addresses used in multiple CoinJoins...")
        assert len(input_used_twice) == 1, f"Expected 1 input address used in multiple CoinJoins, found {len(input_used_twice)}"
        print(f"    ✅ Found {len(input_used_twice)} input address used in multiple CoinJoins") 

        print(f"  ✓ Checking input addresses used in multiple CoinJoins...")
        assert input_used_twice[0][0] == 1, f"Expected input_address_id 1 to be used in multiple CoinJoins, found {input_used_twice[0][0]}"
        print(f"    ✅ Found input_address_id {input_used_twice[0][0]} used in multiple CoinJoins")
        print("\n✅ All assertions passed!")


        print(f"\n✨ Completed processing all transactions!\n")
