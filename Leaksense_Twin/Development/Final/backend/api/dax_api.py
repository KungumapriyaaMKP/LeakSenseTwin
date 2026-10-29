from fastapi import APIRouter, Request
from dax.measures import MEASURES
import time

router = APIRouter(prefix="/api/dax", tags=["DAX Engine"])

@router.post("/query")
async def dax_query(request: Request, body: dict):
    dax_engine = request.app.state.dax
    expr = body.get("expression", "")
    t0 = time.time()
    result = dax_engine.query(expr)
    t1 = time.time()
    return {
        "expression": expr,
        "result": result,
        "execution_ms": round((t1 - t0) * 1000, 2)
    }

@router.get("/measures")
async def list_measures():
    return {"measures": [
        {"name": k, "formula": v, "description": ""}
        for k, v in MEASURES.items()
    ]}

@router.post("/validate")
async def validate_dax(request: Request, body: dict):
    dax_engine = request.app.state.dax
    expr = body.get("expression", "")
    try:
        sql = dax_engine._dax_to_sql(dax_engine._expand_measures(expr))
        dax_engine.conn.execute(f"EXPLAIN {sql}")
        return {"valid": True, "translated_sql": sql}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@router.get("/suggestions")
async def dax_suggestions(prefix: str = ""):
    funcs = ["CALCULATE","FILTER","AVERAGEX","SUMX","COUNTX","MINX","MAXX",
             "IF","DIVIDE","TOPN","DATESBETWEEN","DATEADD","VALUES","ALL",
             "RANKX","CORR","STDEV","PERCENTILE","RELATED","DISTINCTCOUNT"]
    measures = list(MEASURES.keys())
    tables = ["SensorReadings","TwinResiduals","LeakEvents","ThresholdChecks","EngineSpecs"]
    all_items = funcs + [f"[{m}]" for m in measures] + tables
    filtered = [x for x in all_items if prefix.upper() in x.upper()]
    return {"suggestions": filtered[:20]}
