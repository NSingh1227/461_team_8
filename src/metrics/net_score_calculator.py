from typing import Dict, Tuple
import time


class NetScoreCalculator:
    """Calculate weighted net score based on all individual metrics."""
    
    def __init__(self) -> None:
        # Weights based on Sarah's priorities from requirements
        self.weights = {
            'ramp_up_time': 0.25,   
            'license': 0.20,          
            'dataset_and_code_score': 0.15, 
            'bus_factor': 0.15,       
            'performance_claims': 0.10,  
            'code_quality': 0.10,       
            'dataset_quality': 0.05   
        }
    
    def calculate_net_score(self, metrics: Dict[str, float]) -> Tuple[float, int]:
        """
        Calculate weighted net score from all individual metrics.
        """
        start_time = time.time()
        
        net_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                net_score += weight * metrics[metric_name]
                total_weight += weight
        
        if total_weight > 0:
            net_score = net_score / total_weight
        else:
            net_score = 0.5 
        
        net_score = max(0.0, min(1.0, net_score))
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        return net_score, latency_ms
