import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

def create_database_datagathering():
    """Create DuckDB database with script_type enum."""

    # Get the database path
    db_path = os.getenv("CLUSTERING_DATABASE_PATH")

    # Create new database
    print("=" * 80)
    print(f"📊 INITIALIZING DATABASE")
    print("=" * 80)
    print(f"📁 Database path: {db_path}\n")
    conn = duckdb.connect(db_path)

    print("-" * 80)
    print("🔧 STEP 1: Creating Custom Types")
    print("-" * 80)

    try:
        # Create script_pubkey_type enum
        print("🚀 Creating script_pubkey_type enum...")
        conn.execute("""
            CREATE TYPE IF NOT EXISTS script_pubkey_type AS ENUM (
                'p2pkh',
                'p2sh',
                'v0_p2wpkh',
                'v0_p2wsh',
                'v1_p2tr',
                'other'
            )
        """)
        print("✅ script_pubkey_type enum created successfully")

        # Verify enum was created
        result = conn.execute("""
            SELECT enum_range(NULL::script_pubkey_type) AS script_types
        """).fetchone()
        print(f"   📋 Available types: {result[0]}\n")

    except Exception as e:
        print(f"❌ Error creating script_pubkey_type enum: {e}")
        raise

    try:
        # Create transaction_input struct
        print("🚀 Creating transaction_input struct...")
        conn.execute("""
            CREATE TYPE IF NOT EXISTS transaction_input AS STRUCT (
                prev_tx_id BLOB,
                prev_vout_index INTEGER,
                script_pubkey_type script_pubkey_type,
                script_pubkey_address BLOB,
                input_value_satoshi BIGINT,
                is_coinbase BOOL,
                input_address_id INTEGER
            )
        """)
        print("✅ transaction_input struct created successfully")

        # Verify struct was created
        result = conn.execute("""
            SELECT * FROM duckdb_types() WHERE type_name = 'transaction_input'
        """).fetchone()
        if result:
            print(f"   ✓ Verified: {result[0]}\n")

    except Exception as e:
        print(f"❌ Error creating transaction_input struct: {e}")
        raise

    try:
        # Create transaction_output struct
        print("🚀 Creating transaction_output struct...")
        conn.execute("""
            CREATE TYPE IF NOT EXISTS transaction_output AS STRUCT (
                vout_index INTEGER,
                script_pubkey_type script_pubkey_type,
                script_pubkey_address BLOB,
                output_value_satoshi BIGINT,
                output_address_id INTEGER
            )
        """)
        print("✅ transaction_output struct created successfully")

        # Verify struct was created
        result = conn.execute("""
            SELECT * FROM duckdb_types() WHERE type_name = 'transaction_output'
        """).fetchone()
        if result:
            print(f"   ✓ Verified: {result[0]}\n")

    except Exception as e:
        print(f"❌ Error creating transaction_output struct: {e}")
        raise

    print("-" * 80)
    print("🔧 STEP 2: Creating Sequences")
    print("-" * 80)

    try:
        # Create coordinator sequence
        print("🚀 Creating coordinator sequence...")
        conn.execute("CREATE SEQUENCE coor_id_seq START 1;")
        print("✅ Coordinator sequence created successfully\n")

    except Exception as e:
        print(f"❌ Error creating coordinator sequence: {e}")
        raise

    print("-" * 80)
    print("🔧 STEP 3: Creating Tables")
    print("-" * 80)

    try:
        # Create coordinator table
        print("🚀 Creating coordinators table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS coordinators (
                coor_id INTEGER PRIMARY KEY DEFAULT nextval('coor_id_seq'),          -- internal numeric ID
                coordinator_endpoint VARCHAR UNIQUE                 -- URL of coordinator
            )
        """)
        print("✅ Coordinators table created successfully\n")

    except Exception as e:
        print(f"❌ Error creating coordinator table: {e}")
        raise

    try:
        # Create transaction sequence
        print("🚀 Creating transaction sequence...")
        conn.execute("CREATE SEQUENCE tx_id_seq START 1;")
        print("✅ Transaction sequence created successfully\n")

    except Exception as e:
        print(f"❌ Error creating transaction sequence: {e}")
        raise

    try:
        # Create transactions table
        print("🚀 Creating coinjoin_transactions table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS coinjoin_transactions (
                tx_id_int INTEGER PRIMARY KEY DEFAULT nextval('tx_id_seq'),
                tx_id BLOB UNIQUE,                          -- raw 32-byte txid

                number_inputs INTEGER,
                number_outputs INTEGER,
                value_inputs BIGINT,                        -- sum of input_value_satoshi (sats)
                value_outputs BIGINT,                       -- sum of output_value_satoshi (sats)

                inputs transaction_input[],                 -- full vin list
                outputs transaction_output[],               -- full vout list

                coordinator_id INTEGER REFERENCES coordinators(coor_id),

                transaction_fee BIGINT,                     -- sats
                block_number INTEGER,                       -- block height
                block_time TIMESTAMP,

                -- size / weight / feerate
                raw_size_bytes INTEGER,                     -- serialized size in bytes
                weight INTEGER,                             -- weight units
                fee_rate_sat_per_vbyte DOUBLE,              -- fee_rate = fee / vsize

                -- processing status
                processed BOOLEAN NOT NULL DEFAULT FALSE    -- whether transaction has been processed for clustering
            )
        """)
        print("✅ Coinjoin_transactions table created successfully\n")

    except Exception as e:
        print(f"❌ Error creating transactions table: {e}")
        raise

    try:
        # Create input sequence
        print("🚀 Creating input address sequence...")
        conn.execute("CREATE SEQUENCE input_id_seq START 1;")
        print("✅ Input address sequence created successfully\n")

    except Exception as e:
        print(f"❌ Error creating input sequence: {e}")
        raise

    try:
        # Create input_addresses table
        print("🚀 Creating input_addresses table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS input_addresses (
                input_address_id INTEGER PRIMARY KEY DEFAULT nextval('input_id_seq'),
                address BLOB UNIQUE,                        -- scriptpubkey_address in binary form
                used_as_output BOOLEAN,                     -- true if this address also appears as CJ output
                script_type script_pubkey_type,
                number_of_cjs_used_in_as_input INTEGER,
                total_amount_spent_in_cj BIGINT             -- sats
            )
        """)
        print("✅ Input_addresses table created successfully\n")

    except Exception as e:
        print(f"❌ Error creating input_addresses table: {e}")
        raise

    try:
        # Create output sequence
        print("🚀 Creating output address sequence...")
        conn.execute("CREATE SEQUENCE output_id_seq START 1;")
        print("✅ Output address sequence created successfully\n")

    except Exception as e:
        print(f"❌ Error creating output sequence: {e}")
        raise

    try:
        # Create output_addresses table
        print("🚀 Creating output_addresses table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS output_addresses (
                output_address_id INTEGER PRIMARY KEY DEFAULT nextval('output_id_seq'),
                address BLOB UNIQUE,                        -- scriptpubkey_address in binary form
                used_as_input BOOLEAN,                      -- true if this address also appears as CJ input
                script_type script_pubkey_type,
                number_of_cjs_used_in_as_output INTEGER,
                total_amount_received_in_cj BIGINT          -- sats
            )
        """)
        print("✅ Output_addresses table created successfully\n")

    except Exception as e:
        print(f"❌ Error creating output_addresses table: {e}")
        raise

    finally:
        conn.close()
        print("=" * 80)
        print("🎉 DATABASE SETUP COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"📁 Database location: {db_path}\n")

if __name__ == "__main__":
    create_database_datagathering()
