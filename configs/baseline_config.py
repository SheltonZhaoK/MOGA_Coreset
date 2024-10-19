class Arguments:
    def __init__(self):
        self.seed = 1

        self.arguments = {
            "seed": self.seed,
            "test_size": 0.3,

            "Classifiers":
            {   
                    "LR":
                    {
                        'solver':["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                        'C' : [10, 1.0, 0.1, 0.01]
                    }
            }
        }