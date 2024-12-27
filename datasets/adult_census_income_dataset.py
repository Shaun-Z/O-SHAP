import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.neighbors
from datasets.base_dataset import BaseDataset

class AdultCensusIncomeDataset(BaseDataset):
    """
    A dataset class for Adult Census dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(dataroot="./data/adult_census_income")
        parser.set_defaults(num_classes=2)
        return parser

    def __init__(self, opt):
        """
        Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.dtypes = [
            ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
            ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
            ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
            ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
            ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
        ]
        self.data = None
        self.label = None
        self.load_data()

    def load_data(self):
        """
        Load the dataset from the disk.

        Returns:
            data (pd.DataFrame) -- the loaded dataset
        """
        raw_data = pd.read_csv(
            os.path.join(self.root, "adult.csv"),
            header=0,
            names=[d[0] for d in self.dtypes],
            na_values="?",
            dtype=dict(self.dtypes)
        )

        data = raw_data.drop(["Education"], axis=1)  # redundant with Education-Num
        filt_dtypes = list(filter(lambda x: x[0] not in ["Target", "Education"], self.dtypes))
        data["Target"] = data["Target"] == ">50K"
        rcode = {
            "Not-in-family": 0,
            "Unmarried": 1,
            "Other-relative": 2,
            "Own-child": 3,
            "Husband": 4,
            "Wife": 5
        }
        for k, dtype in filt_dtypes:
            if dtype == "category":
                if k == "Relationship":
                    data[k] = np.array([rcode[v.strip()] for v in data[k]])
                else:
                    data[k] = data[k].cat.codes

        X = data.drop(["Target", "fnlwgt"], axis=1)
        y = data["Target"].values

        self.mean = X.mean()
        self.std = X.std()

        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=7)

        X_train = (X_train - self.mean) / self.std
        X_valid = (X_valid - self.mean) / self.std
                
        self.data = X_train if self.phase == "train" else X_valid
        self.label = y_train if self.phase == "train" else y_valid

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index (int) -- a random integer for data indexing

        Returns:
            a dictionary of data with their names.
        """
        data = self.data.iloc[index]
        label = self.label[index]
        return {"X": data, "Y": label}
    
if __name__ == "__main__":
    import shap

    class Option:
        phase = "train"
        dataroot = "./data/adult_census_income"

    opt = Option()
    dataset = AdultCensusIncomeDataset(opt)
    print(len(dataset))
    print(dataset[0])
    X_train = dataset.data
    y_train = dataset.label

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, y_train)

    opt.phase = "val"
    dataset_val = AdultCensusIncomeDataset(opt)
    X_valid = dataset_val.data
    y_valid = dataset_val.label
    def f(x):
        return knn.predict_proba(x)[:, 1]
    
    med = X_train.median().values.reshape((1, X_train.shape[1]))

    explainer = shap.Explainer(f, med)
    shap_values = explainer(X_valid.iloc[0:1000, :])
    shap.plots.waterfall(shap_values[0])
    shap.plots.beeswarm(shap_values)
    shap.plots.heatmap(shap_values)