# Pre-built DAX Measures
MEASURES = {
    "Leak Rate":
        "DIVIDE(COUNTX(FILTER(LeakEvents,[go_no_go]='NO-GO'),1), COUNTX(SensorReadings,1),0)",
    "Avg Confidence When Leaking":
        "AVERAGEX(FILTER(LeakEvents,[confidence]>0),[confidence])",
    "Avg Flow Loss":
        "AVERAGEX(FILTER(LeakEvents,[flow_loss_pct]>0),[flow_loss_pct])",
    "Peak EF Deviation":
        "MAXX(TwinResiduals,[ef_deviation])",
    "Detection Accuracy":
        "DIVIDE(COUNTX(FILTER(LeakEvents,[confidence]>=0.85),1), COUNTX(LeakEvents,1),0)",
    "MAF Deviation Pct":
        "DIVIDE(ABS(AVERAGE(SensorReadings[maf])-842),842,0)*100",
    "Boost Deviation Pct":
        "DIVIDE(ABS(AVERAGE(SensorReadings[map_boost])-218),218,0)*100",
    "Avg CAC Effectiveness":
        "1 - DIVIDE(AVERAGE(SensorReadings[t_cac_out])-25, CALCULATE(AVERAGE([t_boost]),ALL())-25,0)",
    "Steady State Coverage":
        "DIVIDE(COUNTX(FILTER(SensorReadings,[is_steady_state]=1),1), COUNTX(SensorReadings,1),0)",
    "Most Common Leak Zone":
        "FIRSTNONBLANK(TOPN(1,VALUES(LeakEvents[zone]), CALCULATE(COUNTX(LeakEvents,1)),DESC),[zone])",
    "Zone B Leak Count":
        "COUNTX(FILTER(LeakEvents,[zone]='B'),1)",
    "Zone B Avg Confidence":
        "AVERAGEX(FILTER(LeakEvents,[zone]='B'),[confidence])",
    "MTTD Minutes":
        "AVERAGEX(LeakEvents, DATEDIFF([session_start],[detected_at],MINUTE))",
    "Leak Events Today":
        "COUNTX(FILTER(LeakEvents, DATESBETWEEN([detected_at],TODAY(),TODAY()+1)),1)",
    "Rolling 1Hr Leak Rate":
        "DIVIDE(COUNTX(FILTER(LeakEvents, [detected_at]>=DATEADD(NOW(),-1,'hour')),1), COUNTX(SensorReadings,1)*3600,0)",
    "Max MAF Residual Pct":
        "MAXX(TwinResiduals,ABS(DIVIDE([res_maf],842,0)))*100",
    "Avg Cosine Similarity":
        "AVERAGEX(FILTER(TwinResiduals,[ef_deviation]>0),[cosine_sim])",
    "Threshold Check Count Today":
        "COUNTX(FILTER(ThresholdChecks, DATESBETWEEN([checked_at],TODAY(),TODAY()+1)),1)",
    "Inference Latency P95":
        "PERCENTILE(LeakEvents,0.95,[processing_ms])",
    "System Uptime Pct":
        "DIVIDE(COUNTX(FILTER(SensorReadings,[is_steady_state]=1),1), COUNTX(SensorReadings,1),0)*100",
}
