class Rewards:
    @staticmethod
    def calculate_job_reward(success: bool, t_rec: float, deadline: float, lambda_t: float = 1.0) -> float:
        """
        Calculates the application-tied reward for a resolved job.
        r = S - lambda_T * min(T_rec / tau, 1)
        """
        if not success:
            return 0.0
            
        # Environment deadlines are stored in milliseconds while t_rec is in
        # seconds. Convert once here so the frozen objective uses tau=0.1 s.
        deadline_s = deadline / 1000.0
        latency_penalty = min(t_rec / deadline_s, 1.0)
        return 1.0 - lambda_t * latency_penalty
        
    @staticmethod
    def aggregate_epoch_reward(job_rewards: list) -> float:
        if not job_rewards:
            return 0.0
        return sum(job_rewards) / len(job_rewards)
