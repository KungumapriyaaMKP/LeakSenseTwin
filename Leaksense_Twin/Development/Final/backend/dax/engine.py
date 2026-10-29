import duckdb
import re
import pandas as pd
import os
from dax.measures import MEASURES

class DAXEngine:
    def __init__(self):
        self.conn = duckdb.connect(":memory:")
        # We will create empty schema to allow registration of measures
        self._create_empty_tables()
        self._load_external_dataset()
        self._register_measures()
        
    def _load_external_dataset(self):
        # The file is currently inside backend/dax/engine.py, so its dir is backend/dax
        # To get to backend/data:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'LeakSense_Twin_Dataset.xlsx')
        dataset_path_v2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'LeakSense_Twin_Dataset_v2.xlsx')
        
        # Load the first one that exists
        try:
            if os.path.exists(dataset_path):
                df_ext = pd.read_excel(dataset_path, header=1)
                self.conn.register('df_ext', df_ext)
                self.conn.execute("CREATE OR REPLACE TABLE LeakSense_Twin_Dataset AS SELECT * FROM df_ext")
                print(f"[DAX] Loaded External Dataset: {dataset_path}")
            elif os.path.exists(dataset_path_v2):
                df_ext = pd.read_excel(dataset_path_v2, header=1)
                self.conn.register('df_ext', df_ext)
                self.conn.execute("CREATE OR REPLACE TABLE LeakSense_Twin_Dataset AS SELECT * FROM df_ext")
                print(f"[DAX] Loaded External Dataset: {dataset_path_v2}")
        except Exception as e:
            print(f"[DAX] Failed to load external dataset: {e}")

    def _create_empty_tables(self):
        # Create empty tables so measures can be registered
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS SensorReadings (
                timestamp VARCHAR, maf DOUBLE, map_boost DOUBLE, t_cac_out DOUBLE, t_boost DOUBLE, is_steady_state INT
            );
            CREATE TABLE IF NOT EXISTS LeakEvents (
                detected_at VARCHAR, timestamp VARCHAR, go_no_go VARCHAR, confidence DOUBLE, 
                flow_loss_pct DOUBLE, zone VARCHAR, session_start VARCHAR, processing_ms DOUBLE
            );
            CREATE TABLE IF NOT EXISTS TwinResiduals (
                timestamp VARCHAR, ef_deviation DOUBLE, res_maf DOUBLE, cosine_sim DOUBLE
            );
            CREATE TABLE IF NOT EXISTS ThresholdChecks (
                checked_at VARCHAR
            );
        ''')

    def refresh_tables(self, history):
        if not history:
            return
            
        # Parse history into flat tabular structures
        readings = []
        events = []
        residuals = []
        for item in history:
            ts = item.get('timestamp')
            sensors = item.get('sensors', {})
            readings.append({
                'timestamp': ts,
                'maf': sensors.get('MAF', 0),
                'map_boost': sensors.get('MAP_boost', 0),
                't_cac_out': sensors.get('T_cac_out', 0),
                't_boost': sensors.get('T_boost', 0),
                'is_steady_state': 1
            })
            
            events.append({
                'detected_at': ts,
                'timestamp': ts,
                'go_no_go': item.get('go_no_go', 'GO'),
                'confidence': item.get('confidence', 0.0),
                'flow_loss_pct': item.get('flow_loss_pct', 0.0),
                'zone': item.get('suspected_zone', ''),
                'session_start': ts,
                'processing_ms': 0.0
            })
            
            res = item.get('residuals', {})
            ef = item.get('energy_field', {})
            residuals.append({
                'timestamp': ts,
                'ef_deviation': ef.get('global_deviation', 0.0),
                'res_maf': res.get('res_MAF', 0.0),
                'cosine_sim': ef.get('cosine_similarity', 1.0)
            })
            
        df_readings = pd.DataFrame(readings)
        df_events = pd.DataFrame(events)
        df_residuals = pd.DataFrame(residuals)
        
        self.conn.register('df_readings', df_readings)
        self.conn.register('df_events', df_events)
        self.conn.register('df_residuals', df_residuals)
        
        self.conn.execute("CREATE OR REPLACE TABLE SensorReadings AS SELECT * FROM df_readings")
        self.conn.execute("CREATE OR REPLACE TABLE LeakEvents AS SELECT * FROM df_events")
        self.conn.execute("CREATE OR REPLACE TABLE TwinResiduals AS SELECT * FROM df_residuals")

    def _register_measures(self):
        for name, formula in MEASURES.items():
            try:
                sql = self._dax_to_sql(formula)
                self.conn.execute(f"CREATE OR REPLACE VIEW [_{name}] AS {sql}")
            except Exception as e:
                pass # Ignore missing tables initially

    def query(self, dax_expression: str, timeout_ms: int = 5000) -> dict:
        try:
            expanded = self._expand_measures(dax_expression)
            sql = self._dax_to_sql(expanded)
            result = self.conn.execute(sql).df()

            # Sanitize: convert numpy types and NaN/Inf to JSON-safe Python natives
            import math
            import numpy as np

            def _clean(v):
                if v is None:
                    return None
                # Convert numpy integer types
                if isinstance(v, (np.integer,)):
                    return int(v)
                # Convert numpy float types
                if isinstance(v, (np.floating,)):
                    if np.isnan(v) or np.isinf(v):
                        return None
                    return float(v)
                # Python float NaN/Inf
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                # Convert numpy bool
                if isinstance(v, (np.bool_,)):
                    return bool(v)
                return v

            result = result.where(pd.notnull(result), other=None)
            rows = [
                {col: _clean(row[col]) for col in result.columns}
                for _, row in result.iterrows()
            ]
            columns = list(result.columns)
            is_scalar = len(result) == 1 and len(columns) == 1
            scalar_value = None
            if is_scalar:
                raw = result.iloc[0, 0]
                cleaned = _clean(raw)
                if cleaned is not None:
                    try:
                        scalar_value = float(cleaned)
                    except (TypeError, ValueError):
                        scalar_value = None

            return {
                "success": True,
                "rows": rows,
                "columns": columns,
                "row_count": int(len(result)),
                "is_scalar": is_scalar,
                "scalar_value": scalar_value,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "rows": [], "columns": []}

    def _expand_measures(self, expr: str) -> str:
        for name, formula in MEASURES.items():
            expr = expr.replace(f"[{name}]", f"({formula})")
        return expr

    def _dax_to_sql(self, dax: str) -> str:
        sql = dax.strip()

        # ── 1. Quote bracketed column/measure references ──────────────────────
        # e.g. [maf] → "maf", [go_no_go] → "go_no_go"
        sql = re.sub(r'\[([^\]]+)\]', r'"\1"', sql)

        # ── 2. TOPN(n, Table, expr, ASC|DESC) ────────────────────────────────
        sql = re.sub(
            r'TOPN\((\d+),\s*(\w+),\s*(.+?),\s*(ASC|DESC)\)',
            r'SELECT * FROM \2 ORDER BY \3 \4 LIMIT \1',
            sql, flags=re.DOTALL
        )

        # ── 3. AVERAGEX with FILTER ───────────────────────────────────────────
        sql = re.sub(
            r'AVERAGEX\(FILTER\((\w+),\s*(.+?)\),\s*(.+?)\)',
            r'SELECT AVG(\3) AS result FROM \1 WHERE \2',
            sql, flags=re.DOTALL
        )

        # ── 4. AVERAGEX without FILTER ────────────────────────────────────────
        sql = re.sub(
            r'AVERAGEX\((\w+),\s*(.+?)\)',
            r'SELECT AVG(\2) AS result FROM \1',
            sql, flags=re.DOTALL
        )

        # ── 5. COUNTX with FILTER ─────────────────────────────────────────────
        sql = re.sub(
            r'COUNTX\(FILTER\((\w+),\s*(.+?)\),\s*\d+\)',
            r'SELECT COUNT(*) AS result FROM \1 WHERE \2',
            sql
        )

        # ── 6. COUNTX without FILTER ──────────────────────────────────────────
        sql = re.sub(r'COUNTX\((\w+),\s*\d+\)', r'SELECT COUNT(*) AS result FROM \1', sql)

        # ── 7. MAXX with FILTER ───────────────────────────────────────────────
        sql = re.sub(
            r'MAXX\(FILTER\((\w+),\s*(.+?)\),\s*(.+?)\)',
            r'SELECT MAX(\3) AS result FROM \1 WHERE \2',
            sql, flags=re.DOTALL
        )

        # ── 8. MAXX without FILTER ────────────────────────────────────────────
        sql = re.sub(
            r'MAXX\((\w+),\s*(.+?)\)',
            r'SELECT MAX(\2) AS result FROM \1',
            sql, flags=re.DOTALL
        )

        # ── 9. MINX with FILTER ───────────────────────────────────────────────
        sql = re.sub(
            r'MINX\(FILTER\((\w+),\s*(.+?)\),\s*(.+?)\)',
            r'SELECT MIN(\3) AS result FROM \1 WHERE \2',
            sql, flags=re.DOTALL
        )

        # ── 10. MINX without FILTER ───────────────────────────────────────────
        sql = re.sub(
            r'MINX\((\w+),\s*(.+?)\)',
            r'SELECT MIN(\2) AS result FROM \1',
            sql, flags=re.DOTALL
        )

        # ── 11. CALCULATE + FILTER ────────────────────────────────────────────
        sql = re.sub(
            r'CALCULATE\((.+?),\s*FILTER\((\w+),\s*(.+?)\)\)',
            r'SELECT \1 AS result FROM \2 WHERE \3',
            sql, flags=re.DOTALL
        )

        # ── 12. AVERAGE(Table[col]) or AVERAGE("col") ─────────────────────────
        sql = re.sub(r'AVERAGE\((\w+\[[\w]+\])\)',
                     lambda m: f'SELECT AVG({m.group(1).split("[")[1].rstrip("]")}) AS result FROM {m.group(1).split("[")[0]}',
                     sql)
        sql = re.sub(r'AVERAGE\("([\w]+)"\)', r'SELECT AVG("\1") AS result FROM SensorReadings', sql)

        # ── 13. STDEV → STDDEV_POP ───────────────────────────────────────────
        sql = sql.replace('STDEV(', 'STDDEV_POP(')

        # ── 14. PERCENTILE ───────────────────────────────────────────────────
        sql = re.sub(
            r'PERCENTILE\((\w+),([\d.]+),(.+?)\)',
            r'SELECT PERCENTILE_CONT(\2) WITHIN GROUP (ORDER BY \3) AS result FROM \1',
            sql
        )

        # ── 15. DIVIDE(num, denom, alt) → CASE WHEN ──────────────────────────
        # Use a function to handle nested expressions properly
        def replace_divide(m):
            num, den, alt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            return f'CASE WHEN ({den})=0 THEN {alt} ELSE ({num})/({den}) END'
        sql = re.sub(r'DIVIDE\((.+?),(.+?),(.+?)\)', replace_divide, sql, flags=re.DOTALL)

        # ── 16. IF → CASE WHEN ───────────────────────────────────────────────
        sql = re.sub(r'IF\((.+?),(.+?),(.+?)\)', r'CASE WHEN \1 THEN \2 ELSE \3 END', sql, flags=re.DOTALL)

        # ── 17. DATEADD ──────────────────────────────────────────────────────
        sql = re.sub(r"DATEADD\((.+?),(-?\d+),'(.+?)'\)", r"(\1 + INTERVAL \2 \3)", sql)

        # ── 18. Wrap bare expressions with SELECT if not already a query ──────
        if not sql.strip().upper().startswith('SELECT'):
            sql = f'SELECT ({sql}) AS result'

        return sql
