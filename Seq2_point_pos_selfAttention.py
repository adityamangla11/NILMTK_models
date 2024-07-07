from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe.shape)
        # print(self.pe[:x.size(1), :].shape, x.shape)
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

# Model definition using self Attention
class Self_Attention2(nn.Module):
    def __init__(self, seqlength, subseqlength):
        super(Self_Attention2, self).__init__()
        self.subseqlength = subseqlength
        self.total_subseq = seqlength // subseqlength
        self.d_model = 50*subseqlength
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=8, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=6, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=4, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=4, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.Conv1d(50, 50, kernel_size=4, stride=1, padding= 'same'),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(self.d_model, 0.1)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.query = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.dens1 = nn.Linear(self.total_subseq*self.d_model, 128)
        self.dens2 = nn.Linear(128, 128)
        self.dens3 = nn.Linear(128, 1)


    def forward(self, x):
        x = x.view(-1, self.total_subseq, self.subseqlength, 1)
        stack = []
        for i in range(x.size(1)):
            stack.append(self.conv_layer(x[:,i,:,:].permute(0, 2, 1)))
        x = torch.stack(stack, dim=1)
        x = x.view(-1, self.total_subseq, self.subseqlength * 50)
#         print(x.shape)
        x = self.pos_encoder(x)
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        x = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        x = torch.nn.functional.softmax(x, dim=-1)
        x = torch.matmul(x, V)
        x = x.view(-1, self.total_subseq*self.d_model)
        x = torch.nn.functional.relu(self.dens1(x))
        x = torch.nn.functional.relu(self.dens2(x))
        x = self.dens3(x)
        return x




class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', verbose=1):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_model_state_dict = None

    def step(self, val_loss, model_state_dict):
        if val_loss < self.best_loss:
            if self.verbose > 0:
                print(f"Validation loss improved ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
            self.best_loss = val_loss
            self.best_model_state_dict = model_state_dict
            torch.save(model_state_dict.state_dict(), self.filepath)

class Seq2Point_selfAttention_pos(Disaggregator):

    def __init__(self, params):
        super(Seq2Point_selfAttention_pos, self).__init__()

        self.MODEL_NAME = "Seq2Point_selfAttention_pos"
        self.models = OrderedDict()
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})
        self.mains_mean = params.get('mains_mean', 1800)
        self.mains_std = params.get('mains_std', 600)
        # if self.sequence_length % 2 == 0:
        #     print("Sequence length should be odd!")
        #     raise SequenceLengthError

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        print("...............Seq2Point_selfAttention_pos partial_fit running...............")
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0)
            app_df_values = app_df.values.reshape((-1, 1))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                print("Started Retraining model for", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                if len(train_main) > 10:
                    filepath = self.file_prefix +".pth"
                    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15, random_state=10)
                    train_dataset = TensorDataset(torch.tensor(train_x).to(device), torch.tensor(train_y).to(device))
                    val_dataset = TensorDataset(torch.tensor(v_x).to(device), torch.tensor(v_y).to(device))
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
                    val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
                    self.train_model(model, train_loader, val_loader, checkpoint)
                    self.models[appliance_name].load_state_dict(torch.load(filepath))

    def train_model(self, model, train_loader, val_loader, checkpoint):
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters())

        for epoch in range(self.n_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)

            if val_loader:
                model.eval()
                val_loss = self.evaluate_model(model, val_loader, criterion)
                print(f"Epoch [{epoch + 1}/{self.n_epochs}], Loss: {epoch_loss}, Val Loss: {val_loss}")
                checkpoint.step(val_loss, model)
            else:
                print(f"Epoch [{epoch + 1}/{self.n_epochs}], Loss: {epoch_loss}")

    def evaluate_model(self, model, val_loader, criterion):
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            disaggregation_dict = {}
            for appliance in self.models:
                model = self.models[appliance]
                model.eval()
                with torch.no_grad():
                    prediction = model(torch.tensor(test_main).to(device))
                prediction = prediction.cpu().numpy()
                prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disaggregation_dict[appliance] = df
            results = pd.DataFrame(disaggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def return_network(self):
        model = Self_Attention2(self.sequence_length, 33).to(device)
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == 'train':
            # Preprocessing for the train data
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print("Parameters for", app_name, "were not found!")
                    raise ApplianceNotFoundError()

                processed_appliance_dfs = []

                for app_df in app_df_list:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    new_app_readings = (new_app_readings - app_mean) / app_std
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_appliance_dfs))
            return mains_df_list, appliance_list

        else:
            # Preprocessing for the test data
            mains_df_list = []

            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def set_appliance_params(self, train_appliances):
        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})
        print(self.appliance_params)
