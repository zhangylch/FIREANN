class Weight_Scheduler():
    def __init__(self,init_weight,final_weight,start_lr,end_lr):
        self.init_weight = init_weight
        self.final_weight = final_weight
        self.start_lr=start_lr
        self.end_lr=end_lr

    def __call__(self,lr):
        return self.init_weight+(self.final_weight-self.init_weight)*(lr-self.start_lr)/(self.end_lr-self.start_lr+1e-8)
