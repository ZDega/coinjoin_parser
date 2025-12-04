from __future__ import annotations

from pathlib import Path
import os
from dotenv import load_dotenv



import duckdb

load_dotenv()

JSON_DIR = os.getenv('RAW_ROUND_JSON_FILES_PATH')
DB_PATH = os.getenv('COINJOIN_RAW_DATA_DATABASE_PATH')



def ensure_table_exists(con:duckdb.DuckDBPyConnection)->bool:
    

    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS raw_round_data (
                    coordinator_endpoint TEXT NOT NULL,
                    estimated_coordinator_earnings_sats INTEGER CHECK(estimated_coordinator_earnings_sats >= 0),
                    round_id BLOB PRIMARY KEY ,
                    isBlame BOOLEAN,
                    coordinaton_fee_rate DOUBLE,
                    min_input_count INTEGER CHECK(min_input_count >= 0),
                    parameters_mining_fee_rate DOUBLE,
                    round_start_time TIMESTAMP,
                    round_end_time TIMESTAMP,
                    tx_id BLOB ,
                    final_mining_fee_rate DOUBLE,
                    virtual_size DOUBLE,
                    total_mining_fee DOUBLE,
                    input_count INTEGER CHECK(input_count >= 0),
                    total_input_amount BIGINT CHECK(total_input_amount >= 0),
                    fresh_inputs_estimate_btc DOUBLE,
                    average_standard_input_anon_set DOUBLE,
                    output_count INTEGER CHECK(output_count >= 0),
                    total_output_amount BIGINT CHECK(total_output_amount >= 0),
                    change_output_ratio DOUBLE,
                    average_standard_output_anon_set DOUBLE,
                    total_left_overs INTEGER CHECK(total_left_overs >= 0),
                    processed BOOLEAN NOT NULL DEFAULT FALSE
                     )
                    """)
        return True
    except:
        return False

def fill_data_base(con:duckdb.DuckDBPyConnection, data_file:str)->bool:

    # Resolve: two directories above the script folder -> CoinJoin_data
    data_file_path = Path(data_file)
    
 
    
    con.execute(f"""
        INSERT OR IGNORE INTO raw_round_data (
            coordinator_endpoint, estimated_coordinator_earnings_sats, round_id, isBlame,
            coordinaton_fee_rate, min_input_count, parameters_mining_fee_rate,
            round_start_time, round_end_time, tx_id, final_mining_fee_rate, virtual_size,
            total_mining_fee, input_count, total_input_amount, fresh_inputs_estimate_btc,
            average_standard_input_anon_set, output_count, total_output_amount,
            change_output_ratio, average_standard_output_anon_set, total_left_overs
            )
        SELECT
            CoordinatorEndpoint              AS coordinator_endpoint,
            EstimatedCoordinatorEarningsSats AS estimated_coordinator_earnings_sats,
            from_hex(RoundId)                AS round_id,
            IsBlame                          AS isBlame,
            CoordinationFeeRate              AS coordinaton_fee_rate,
            MinInputCount                    AS min_input_count,
            ParametersMiningFeeRate          AS parameters_mining_fee_rate,
            RoundStartTime                   AS round_start_time,
            RoundEndTime                     AS round_end_time,
            from_hex(TxId)                   AS tx_id,
            FinalMiningFeeRate               AS final_mining_fee_rate,
            VirtualSize                      AS virtual_size,
            TotalMiningFee                   AS total_mining_fee,
            InputCount                       AS input_count,
            TotalInputAmount                 AS total_input_amount,
            FreshInputsEstimateBtc           AS fresh_inputs_estimate_btc,
            AverageStandardInputsAnonSet     AS average_standard_input_anon_set,
            OutputCount                      AS output_count,
            TotalOutputAmount                AS total_output_amount,
            ChangeOutputsAmountRatio         AS change_output_ratio,
            AverageStandardOutputsAnonSet    AS average_standard_output_anon_set,
            TotalLeftOvers                   AS total_left_overs

        FROM read_json_auto('{data_file_path.as_posix()}/*.json');
        """)
    
    

    return True



if __name__ == "__main__":
    # only run when you execute this file directly: `python inset_raw_round_data_in_db.py`
    print(DB_PATH)
    con = duckdb.connect(DB_PATH)
    print(ensure_table_exists(con))
    print(fill_data_base(con,JSON_DIR))


