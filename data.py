from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data import *

class Data:
    def __init__(self):
        pass

    def get_data(self, dataName, configs, args):
        if configs["inputDir"] == "../data/ligand_discovery":
            index = "Catalog Number"
            label = "label"
            data = pd.read_csv(os.path.join(configs["inputDir"], dataName, f"{dataName}_clean_selected_data.csv"), index_col=index)

            train_data, test_data, train_labels, test_labels = train_test_split(data, data[label], test_size=configs["test_size"], random_state=configs["seed"], stratify=data[label].to_list(), shuffle=True) #, shuffle=True
            
            train_labels = train_data[[label]]
            train_data = train_data.drop([label], axis=1)
            
            test_labels = test_data[[label]]
            test_data = test_data.drop([label], axis=1)
            return train_data, train_labels, test_data, test_labels

        elif configs["inputDir"] == "../data/cellLine_integration":
            assert not args.i == "None", "need -i argument for reference data to specify which integration to use"
            label = "CELL_LINE"
            data_dir = configs["inputDir"]
            data = pd.read_csv(os.path.join(data_dir, dataName, f"{args.i}_umap.csv"), index_col = "Unnamed: 0")
            labels = pd.read_csv(os.path.join(data_dir, dataName, f"{args.i}_labels.csv"), index_col = "Unnamed: 0")[[label]]
            labels[label] = LabelEncoder().fit_transform(labels[label])
            labels = labels.rename(columns={label: 'label'})

            # if "10X" in dataName:
            #     # Combine data and labels for consistent downsizing
            #     combined = pd.concat([data, labels], axis=1)
            #     combined = combined.sample(frac=0.1, random_state=configs["seed"])
            #     data = combined.iloc[:, :-1]
            #     labels = combined.iloc[:, -1:]

            train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=configs["test_size"], random_state=configs["seed"], stratify=labels["label"].to_list(), shuffle=True)
            return train_data, train_labels, test_data, test_labels

        elif configs["inputDir"] == "/deac/csc/khuriGrp/wangy22/edcs/data":
            label = "label"
            data_dir = configs["inputDir"]
            data = pd.read_csv(os.path.join(data_dir, dataName), index_col = "CERAPP_ID")
            
            train_labels = data[[label]]
            train_data = data.drop([label], axis=1)

            return train_data, train_labels, None, None