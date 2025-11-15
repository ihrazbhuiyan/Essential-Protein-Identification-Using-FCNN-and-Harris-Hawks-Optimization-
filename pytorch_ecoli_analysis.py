import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import random
import time
from datetime import datetime
import gc
import psutil
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ProteinClassifier(nn.Module):
    def __init__(self, input_shapes, params):
        super(ProteinClassifier, self).__init__()

        # Process embedding input
        self.emb_layer = nn.Sequential(
            nn.Linear(input_shapes['emb'][0], params['dense1']),
            nn.ReLU(),
            nn.Dropout(params['dropout1'])
        )

        # Process matrix input
        self.matrix_layer = nn.Sequential(
            nn.Linear(input_shapes['matrix'][0] * input_shapes['matrix'][1], params['dense2']),
            nn.ReLU(),
            nn.Dropout(params['dropout2'])
        )

        # Process localization input
        self.loc_layer = nn.Sequential(
            nn.Linear(input_shapes['loc'][0], params['dense3']),
            nn.ReLU(),
            nn.Dropout(params['dropout3'])
        )

        # Merged layers
        merged_size = params['dense1'] + params['dense2'] + params['dense3']
        self.merged_layer = nn.Sequential(
            nn.Linear(merged_size, params['dense_merged']),
            nn.ReLU(),
            nn.Dropout(params['dropout_merged']),
            nn.Linear(params['dense_merged'], 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_emb, x_matrix, x_loc):
        # Process each input branch
        x1 = self.emb_layer(x_emb)
        
        batch_size = x_matrix.size(0)
        x_matrix_flat = x_matrix.view(batch_size, -1)
        x2 = self.matrix_layer(x_matrix_flat)
        
        x3 = self.loc_layer(x_loc)
        
        # Concatenate features
        merged = torch.cat((x1, x2, x3), dim=1)
        
        # Final processing
        output = self.merged_layer(merged)
        return output


class HHOProteinClassifier:
    def __init__(self, input_shapes, num_hawks=30, num_iterations=1, model_path='protein_model.pt',
                 continue_training=False, previous_best_solution=None, checkpoint_dir='checkpoints'):
        self.num_hawks = num_hawks
        self.num_iterations = num_iterations
        self.model_path = model_path
        self.input_shapes = input_shapes
        self.continue_training = continue_training
        self.best_solution = previous_best_solution
        self.best_fitness = float('-inf')
        self.iteration_times = []
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def print_memory_usage(self):
        """Monitor system memory usage"""
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    def save_checkpoint(self, iteration, population, fitness_values):
        """Save checkpoint with current state"""
        checkpoint = {
            'iteration': iteration,
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'population': population,
            'fitness_values': fitness_values,
            'iteration_times': self.iteration_times
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_iter_{iteration}.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, cls=NumpyEncoder)

    def load_checkpoint(self):
        """Load the latest checkpoint if it exists"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('checkpoint_iter_')]
        if not checkpoints:
            return None
        
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
        
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint



    def create_model(self, params):
        model = ProteinClassifier(self.input_shapes, params).to(self.device)
        return model

    def initialize_population(self):
        if self.continue_training and self.best_solution:
            population = [self.best_solution]
            for _ in range(self.num_hawks - 1):
                hawk = {}
                for param, value in self.best_solution.items():
                    if isinstance(value, float):
                        hawk[param] = max(0.2, min(0.5, value + random.uniform(-0.1, 0.1)))
                    else:
                        variation = int(value * random.uniform(-0.2, 0.2))
                        # Reduced model capacity
                        max_val = 128 if param == 'dense_merged' else 64
                        hawk[param] = max(16, min(max_val, value + variation))
                population.append(hawk)
        else:
            population = []
            for _ in range(self.num_hawks):
                hawk = {
                    'dense1': random.randint(32, 64),  # Reduced from 32-128
                    'dense2': random.randint(32, 64),  # Reduced from 32-128
                    'dense3': random.randint(16, 64),  # Kept the same
                    'dense_merged': random.randint(32, 128),  # Reduced from 64-256
                    'dropout1': random.uniform(0.2, 0.5),
                    'dropout2': random.uniform(0.2, 0.5),
                    'dropout3': random.uniform(0.2, 0.5),
                    'dropout_merged': random.uniform(0.2, 0.5)
                }
                population.append(hawk)
        return population

    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs_emb, inputs_matrix, inputs_loc, targets in train_loader:
            inputs_emb, inputs_matrix, inputs_loc = inputs_emb.to(self.device), inputs_matrix.to(self.device), inputs_loc.to(self.device)
            targets = targets.to(self.device).float().view(-1, 1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs_emb, inputs_matrix, inputs_loc)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc

    def validate(self, model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs_emb, inputs_matrix, inputs_loc, targets in val_loader:
                inputs_emb, inputs_matrix, inputs_loc = inputs_emb.to(self.device), inputs_matrix.to(self.device), inputs_loc.to(self.device)
                targets = targets.to(self.device).float().view(-1, 1)
                
                # Forward pass
                outputs = model(inputs_emb, inputs_matrix, inputs_loc)
                loss = criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc

    def create_data_loaders(self, X_train, y_train, X_val, y_val, batch_size=64):
        # Convert numpy arrays to PyTorch tensors
        X_train_emb = torch.FloatTensor(X_train[0])
        X_train_matrix = torch.FloatTensor(X_train[1])
        X_train_loc = torch.FloatTensor(X_train[2])
        y_train = torch.FloatTensor(y_train)
        
        X_val_emb = torch.FloatTensor(X_val[0])
        X_val_matrix = torch.FloatTensor(X_val[1])
        X_val_loc = torch.FloatTensor(X_val[2])
        y_val = torch.FloatTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_emb, X_train_matrix, X_train_loc, y_train)
        val_dataset = TensorDataset(X_val_emb, X_val_matrix, X_val_loc, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader


    def fitness_function(self, hawk, X_train, y_train, X_val, y_val):
        try:
            start_time = time.time()
            model = self.create_model(hawk)
            
            # Setup criterion and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)
            
            # Early stopping parameters
            patience = 20
            min_val_loss = float('inf')
            patience_counter = 0
            best_val_acc = 0.0
            
            # Training loop with early stopping
            for epoch in range(5):  # Initial 5 epochs, same as in original code
                # Train
                train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
                
                # Validate
                val_loss, val_acc = self.validate(model, val_loader, criterion)
                
                # Update best validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                # Early stopping
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            duration = time.time() - start_time
            return best_val_acc, duration
        
        finally:
            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def update_hawk_position(self, hawk, rabbit_position, iteration):
        E0 = 0.1
        E = 2 * E0 * (1 - (iteration / self.num_iterations))

        new_hawk = hawk.copy()
        for param in hawk:
            if isinstance(hawk[param], float):
                r = random.random()
                if r >= 0.5:
                    new_hawk[param] = hawk[param] + random.uniform(-1, 1) * E
                    new_hawk[param] = max(0.2, min(0.5, new_hawk[param]))
            else:
                r = random.random()
                if r >= 0.5:
                    new_hawk[param] = int(hawk[param] + random.uniform(-1, 1) * E * 25)
                    if 'dense' in param:
                        max_val = 256 if param == 'dense_merged' else 128
                        new_hawk[param] = max(16, min(max_val, new_hawk[param]))
        return new_hawk

    def train(self, X_train, y_train, X_val, y_val):
        if os.path.exists(self.model_path) and not self.continue_training:
            print("Loading existing model...")
            model = self.create_model(self.best_solution)
            model.load_state_dict(torch.load(self.model_path))
            return model

        if self.continue_training:
            print("Continuing training from previous best solution...")
            if os.path.exists(self.model_path) and self.best_solution:
                model = self.create_model(self.best_solution)
                model.load_state_dict(torch.load(self.model_path))
                base_model = model
            else:
                print("No existing model found. Starting from scratch...")
                base_model = None

        # Try to load checkpoint if continuing training
        start_iteration = 0
        if self.continue_training:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                print(f"Resuming from iteration {checkpoint['iteration']}")
                start_iteration = checkpoint['iteration']
                self.best_solution = checkpoint['best_solution']
                self.best_fitness = checkpoint['best_fitness']
                self.iteration_times = checkpoint.get('iteration_times', [])
                population = checkpoint['population']
            else:
                population = self.initialize_population()
        else:
            population = self.initialize_population()

        start_time = time.time()

        try:
            for iteration in range(start_iteration, self.num_iterations):
                iteration_start = time.time()
                print(f"\nIteration {iteration + 1}/{self.num_iterations}")
                print(f"Start time: {datetime.fromtimestamp(iteration_start).strftime('%Y-%m-%d %H:%M:%S')}")
                self.print_memory_usage()

                # Evaluate fitness for all hawks
                fitness_values = []
                hawk_times = []

                for hawk_idx, hawk in enumerate(population):
                    fitness, duration = self.fitness_function(hawk, X_train, y_train, X_val, y_val)
                    fitness_values.append(fitness)
                    hawk_times.append(duration)

                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = hawk.copy()
                        print(f"New best fitness: {self.best_fitness}")
                        print("Best parameters:", self.best_solution)

                    print(f"Hawk {hawk_idx + 1}/{self.num_hawks}:")
                    print(f"  Training time: {duration:.2f}s")
                    print(f"  Fitness: {fitness:.4f}")

                # Save checkpoint every 5 iterations
                if (iteration + 1) % 5 == 0:
                    self.save_checkpoint(iteration, population, fitness_values)

                # Update hawk positions
                new_population = []
                for hawk in population:
                    new_hawk = self.update_hawk_position(hawk, self.best_solution, iteration)
                    new_population.append(new_hawk)

                population = new_population
                
                iteration_end = time.time()
                iteration_time = iteration_end - iteration_start
                self.iteration_times.append(iteration_time)
                
                print(f"Iteration time: {iteration_time:.2f} seconds")
                print(f"Average iteration time: {np.mean(self.iteration_times):.2f} seconds")
                print(f"Estimated time remaining: {(self.num_iterations - iteration - 1) * np.mean(self.iteration_times):.2f} seconds")

                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.print_memory_usage()

            # Train final model with best parameters
            print("\nTraining final model with best parameters...")
            final_model = self.create_model(self.best_solution)
            
            # Training parameters
            criterion = nn.BCELoss()
            optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
            
            # Create data loaders for final training
            train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)
            
            # Final training loop
            best_val_acc = 0
            best_model_state = None
            
            for epoch in range(100):  # Same as in original code
                # Train
                train_loss, train_acc = self.train_epoch(final_model, train_loader, criterion, optimizer)
                
                # Validate
                val_loss, val_acc = self.validate(final_model, val_loader, criterion)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = final_model.state_dict().copy()
                
                # Print progress
                if epoch % 1 == 0:#if want fewer datapoints div by 10, 5, 2
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Restore best model
            if best_model_state:
                final_model.load_state_dict(best_model_state)
            
            # Save the model and best parameters
            torch.save(final_model.state_dict(), self.model_path)
            np.save(self.model_path.replace('.pt', '_best_params.npy'), self.best_solution)
            
            total_time = time.time() - start_time
            print(f"\nTotal training time: {total_time:.2f} seconds")
            return final_model

        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Save checkpoint on error
            self.save_checkpoint(iteration, population, fitness_values)
            raise


import numpy as np
from scipy.ndimage import gaussian_filter


def augment_protein_data(X_emb, X_matrix, X_loc, y, essential_class=1, augmentation_factor=2):
    """
    Augment protein data for the essential class.

    Parameters:
    -----------
    X_emb : numpy.ndarray
        Protein embedding features
    X_matrix : numpy.ndarray
        Protein matrix features
    X_loc : numpy.ndarray
        Protein localization features
    y : numpy.ndarray
        Class labels (0 for non-essential, 1 for essential)
    essential_class : int, default=1
        The class label for essential proteins
    augmentation_factor : int, default=2
        Number of augmented samples to generate for each essential protein

    Returns:
    --------
    tuple
        (X_emb_augmented, X_matrix_augmented, X_loc_augmented, y_augmented)
    """
    # Find indices of essential proteins
    essential_indices = np.where(y == essential_class)[0]

    # Initialize arrays to store augmented data
    X_emb_aug = []
    X_matrix_aug = []
    X_loc_aug = []
    y_aug = []

    # Add original data
    X_emb_aug.append(X_emb)
    X_matrix_aug.append(X_matrix)
    X_loc_aug.append(X_loc)
    y_aug.append(y)

    # Generate augmented samples for essential proteins
    for _ in range(augmentation_factor):
        # Embeddings augmentation
        emb_aug = augment_embeddings(X_emb[essential_indices])

        # Matrix augmentation
        matrix_aug = augment_matrices(X_matrix[essential_indices])

        # Localization augmentation
        loc_aug = augment_localization(X_loc[essential_indices])

        # Add augmented data
        X_emb_aug.append(emb_aug)
        X_matrix_aug.append(matrix_aug)
        X_loc_aug.append(loc_aug)
        y_aug.append(np.full(len(essential_indices), essential_class))

    # Concatenate original and augmented data
    X_emb_augmented = np.vstack(X_emb_aug)
    X_matrix_augmented = np.vstack(X_matrix_aug)
    X_loc_augmented = np.vstack(X_loc_aug)
    y_augmented = np.hstack(y_aug)

    return X_emb_augmented, X_matrix_augmented, X_loc_augmented, y_augmented


def augment_embeddings(embeddings):
    """
    Augment protein embeddings using jittering and scaling.

    Parameters:
    -----------
    embeddings : numpy.ndarray
        Protein embedding features

    Returns:
    --------
    numpy.ndarray
        Augmented embeddings
    """
    aug_embeddings = embeddings.copy()

    # Apply random jittering (adding small random noise)
    noise_scale = 0.05
    aug_embeddings += np.random.normal(0, noise_scale, aug_embeddings.shape)

    # Apply random scaling (multiplying by a factor close to 1)
    scale_factor = np.random.uniform(0.95, 1.05, size=aug_embeddings.shape)
    aug_embeddings *= scale_factor

    return aug_embeddings


def augment_matrices(matrices):
    """
    Augment protein matrices using small rotations and perturbations.

    Parameters:
    -----------
    matrices : numpy.ndarray
        Protein matrix features

    Returns:
    --------
    numpy.ndarray
        Augmented matrices
    """
    aug_matrices = matrices.copy()
    num_matrices = aug_matrices.shape[0]

    for i in range(num_matrices):
        # Apply one of several augmentation methods randomly
        method = np.random.choice(['blur', 'noise', 'swap', 'none'])

        if method == 'blur':
            # Apply slight Gaussian blur
            aug_matrices[i] = gaussian_filter(aug_matrices[i], sigma=0.5)

        elif method == 'noise':
            # Add small random noise
            noise = np.random.normal(0, 0.03, aug_matrices[i].shape)
            aug_matrices[i] += noise
            # Clip values to maintain valid ranges if necessary
            aug_matrices[i] = np.clip(aug_matrices[i], 0, 1)

        elif method == 'swap':
            # Randomly swap some adjacent elements (simulating minor contact changes)
            if np.random.random() < 0.5 and aug_matrices[i].shape[0] > 1:
                idx1, idx2 = np.random.choice(aug_matrices[i].shape[0], size=2, replace=False)
                aug_matrices[i, [idx1, idx2]] = aug_matrices[i, [idx2, idx1]]

    return aug_matrices


def augment_localization(localization):
    """
    Augment protein localization features.

    Parameters:
    -----------
    localization : numpy.ndarray
        Protein localization features

    Returns:
    --------
    numpy.ndarray
        Augmented localization features
    """
    aug_localization = localization.copy()

    # For binary localization features, occasionally flip less-important locations
    # This assumes localization is one-hot encoded or represents probabilities
    if np.all(np.logical_or(aug_localization == 0, aug_localization == 1)):
        # For binary features, randomly flip with small probability
        # Identify low-confidence or secondary locations to potentially modify
        for i in range(len(aug_localization)):
            # Find indices of 0s (non-locations)
            zero_indices = np.where(aug_localization[i] == 0)[0]
            # Find indices of 1s (locations)
            one_indices = np.where(aug_localization[i] == 1)[0]

            # Occasionally add a secondary location (with 10% probability)
            if len(zero_indices) > 0 and np.random.random() < 0.1:
                random_idx = np.random.choice(zero_indices)
                aug_localization[i, random_idx] = 1

            # Ensure we don't remove all locations
            if len(one_indices) > 1 and np.random.random() < 0.1:
                random_idx = np.random.choice(one_indices)
                aug_localization[i, random_idx] = 0
    else:
        # For continuous features, add small random noise
        noise = np.random.normal(0, 0.05, aug_localization.shape)
        aug_localization += noise
        aug_localization = np.clip(aug_localization, 0, 1)

    return aug_localization

# def load_and_preprocess_data(use_augmentation=True, augmentation_factor=2):
#     # Load data
#     data_dir = 'data'
#     protein_emb = np.load(os.path.join(data_dir, 'protein_emb.npy'))
#     protein_matrix = np.load(os.path.join(data_dir, 'protein_matrix.npy'))
#     protein_loc = np.load(os.path.join(data_dir, 'protein_loc.npy'))
#     protein_label = np.load(os.path.join(data_dir, 'protein_label.npy'))
#
#     protein_label = protein_label.flatten().astype(np.int32)
#
#     print(f"\nData shapes before processing:")
#     print(f"Protein embeddings: {protein_emb.shape}")
#     print(f"Protein matrix: {protein_matrix.shape}")
#     print(f"Protein localization: {protein_loc.shape}")
#     print(f"Protein labels: {protein_label.shape}")
#     print(f"Initial class distribution: {np.bincount(protein_label)}")
#
#     indices = np.arange(len(protein_label))
#     X_train_idx, X_test_idx = train_test_split(indices, test_size=0.1, random_state=42)
#
#     X_train_emb = protein_emb[X_train_idx]
#     X_train_matrix = protein_matrix[X_train_idx]
#     X_train_loc = protein_loc[X_train_idx]
#     y_train = protein_label[X_train_idx]
#
#     # Apply data augmentation before SMOTE if enabled
#     if use_augmentation:
#         print("\nApplying data augmentation for the essential class...")
#         X_train_emb, X_train_matrix, X_train_loc, y_train = augment_protein_data(
#             X_train_emb, X_train_matrix, X_train_loc, y_train,
#             essential_class=1,
#             augmentation_factor=augmentation_factor
#         )
#         print(f"Class distribution after augmentation: {np.bincount(y_train)}")
#
#     print("\nApplying SMOTE for class balance...")
#     X_train_matrix_flat = X_train_matrix.reshape(X_train_matrix.shape[0], -1)
#
#     X_train_combined = np.hstack([
#         X_train_emb,
#         X_train_matrix_flat,
#         X_train_loc
#     ])
#
#     smote = SMOTE(random_state=42)
#     X_train_combined_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)
#
#     emb_size = X_train_emb.shape[1]
#     matrix_size = X_train_matrix.shape[1] * X_train_matrix.shape[2]
#
#     X_train_emb_balanced = X_train_combined_balanced[:, :emb_size]
#     X_train_matrix_balanced = X_train_combined_balanced[:, emb_size:emb_size + matrix_size].reshape(-1,
#                                                                                                    X_train_matrix.shape[1],
#                                                                                                    X_train_matrix.shape[2])
#     X_train_loc_balanced = X_train_combined_balanced[:, emb_size + matrix_size:]
#
#     X_test = [
#         protein_emb[X_test_idx],
#         protein_matrix[X_test_idx],
#         protein_loc[X_test_idx]
#     ]
#     y_test = protein_label[X_test_idx]
#
#     return [X_train_emb_balanced, X_train_matrix_balanced,
#             X_train_loc_balanced], y_train_balanced, X_test, y_test, protein_loc.shape


#E_coli_version
def load_and_preprocess_data(use_augmentation=True, augmentation_factor=2):
    """
    Load preprocessed E. coli data from train/test directories
    """
    print("Loading preprocessed E. coli data from NPY files...")

    # Load training data
    X_train_emb = np.load('data/train/protein_emb.npy')
    X_train_matrix = np.load('data/train/protein_matrix.npy')
    X_train_loc = np.load('data/train/protein_loc.npy')
    y_train = np.load('data/train/protein_label.npy')

    # Load test data
    X_test_emb = np.load('data/test/protein_emb.npy')
    X_test_matrix = np.load('data/test/protein_matrix.npy')
    X_test_loc = np.load('data/test/protein_loc.npy')
    y_test = np.load('data/test/protein_label.npy')

    print(f"\nTraining data shapes:")
    print(f"Embeddings: {X_train_emb.shape}")
    print(f"Matrix: {X_train_matrix.shape}")
    print(f"Localization: {X_train_loc.shape}")
    print(f"Labels: {y_train.shape}")
    print(f"Training class distribution: {np.bincount(y_train)}")

    print(f"\nTest data shapes:")
    print(f"Embeddings: {X_test_emb.shape}")
    print(f"Matrix: {X_test_matrix.shape}")
    print(f"Localization: {X_test_loc.shape}")
    print(f"Labels: {y_test.shape}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    # Apply data augmentation if enabled
    if use_augmentation:
        print("\nApplying data augmentation for the minority class...")
        X_train_emb, X_train_matrix, X_train_loc, y_train = augment_protein_data(
            X_train_emb, X_train_matrix, X_train_loc, y_train,
            essential_class=1,  # Use the positive class
            augmentation_factor=augmentation_factor
        )
        print(f"Training class distribution after augmentation: {np.bincount(y_train)}")

    print("\nApplying SMOTE for class balance...")
    # Flatten matrix for SMOTE
    X_train_matrix_flat = X_train_matrix.reshape(X_train_matrix.shape[0], -1)

    # Combine all features for SMOTE
    X_train_combined = np.hstack([
        X_train_emb,
        X_train_matrix_flat,
        X_train_loc
    ])

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_combined_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)

    # Split back into separate feature types
    emb_size = X_train_emb.shape[1]
    matrix_size = X_train_matrix.shape[1] * X_train_matrix.shape[2]
    loc_size = X_train_loc.shape[1]

    X_train_emb_balanced = X_train_combined_balanced[:, :emb_size]
    X_train_matrix_balanced = X_train_combined_balanced[:,
                              emb_size:emb_size + matrix_size].reshape(-1,
                                                                       X_train_matrix.shape[1],
                                                                       X_train_matrix.shape[2])
    X_train_loc_balanced = X_train_combined_balanced[:, emb_size + matrix_size:]

    # Prepare test data in the expected format
    X_test = [
        X_test_emb,
        X_test_matrix,
        X_test_loc
    ]

    return [X_train_emb_balanced, X_train_matrix_balanced, X_train_loc_balanced], \
        y_train_balanced, X_test, y_test, X_train_loc.shape[1]


def evaluate_model(model, X_test, y_test, device):
    print("\n============= Model Evaluation =============")
    eval_start = time.time()
    
    # Convert test data to PyTorch tensors
    X_test_emb = torch.FloatTensor(X_test[0]).to(device)
    X_test_matrix = torch.FloatTensor(X_test[1]).to(device)
    X_test_loc = torch.FloatTensor(X_test[2]).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Prediction
    with torch.no_grad():
        y_pred_proba = model(X_test_emb, X_test_matrix, X_test_loc)
        y_pred_proba = y_pred_proba.cpu().numpy().flatten()
    
    y_pred = (y_pred_proba > 0.4).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"AUC: {auc:.6f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-essential', 'Essential']))
    
    eval_time = time.time() - eval_start
    print(f"\nEvaluation time: {eval_time:.2f} seconds")

#ecoli_version
def main():
    start_time = time.time()

    print("Starting E. coli classification pipeline...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load preprocessed E. coli data
    X_train, y_train, X_test, y_test, loc_shape = load_and_preprocess_data(
        use_augmentation=True,
        augmentation_factor=2
    )

    # Define input shapes based on your preprocessed data
    input_shapes = {
        'emb': (X_train[0].shape[1],),  # Dynamic based on your data
        'matrix': (X_train[1].shape[1], X_train[1].shape[2]),  # Dynamic
        'loc': (X_train[2].shape[1],)  # Dynamic
    }

    print(f"\nModel input shapes:")
    print(f"Embeddings: {input_shapes['emb']}")
    print(f"Matrix: {input_shapes['matrix']}")
    print(f"Localization: {input_shapes['loc']}")

    # Model path
    model_path = 'ecoli_model.pt'
    best_params_path = model_path.replace('.pt', '_best_params.npy')
    previous_best_solution = None

    # Load previous best parameters if they exist
    if os.path.exists(best_params_path):
        try:
            previous_best_solution = np.load(best_params_path, allow_pickle=True).item()
            print("Loaded previous best parameters:", previous_best_solution)
        except:
            print("Could not load previous best parameters")

    # Initialize and train HHO classifier
    hho_classifier = HHOProteinClassifier(
        input_shapes,
        num_hawks=20,  # Reduced for faster training on smaller dataset
        num_iterations=100,  # Reduced iterations
        model_path=model_path,
        continue_training=False,  # Start fresh for E. coli
        previous_best_solution=previous_best_solution
    )

    # Train model
    model = hho_classifier.train(X_train, y_train, X_test, y_test)

    # Evaluate model
    evaluate_model(model, X_test, y_test, hho_classifier.device)

    total_time = time.time() - start_time
    print(f"\nTotal pipeline execution time: {total_time:.2f} seconds")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# def main():
#     start_time = time.time()
#
#     print("Starting protein classification pipeline...")
#     print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#
#     # Set random seeds for reproducibility
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#
#     # Load and preprocess data with augmentation
#     X_train, y_train, X_test, y_test, loc_shape = load_and_preprocess_data(
#         use_augmentation=True,  # Enable augmentation
#         augmentation_factor=2  # Generate 2 augmented samples for each essential protein
#     )
#
#     # Define input shapes
#     input_shapes = {
#         'emb': (64,),
#         'matrix': (3, 12),
#         'loc': (11,)
#     }
#
#     # Load previous best parameters if they exist
#     model_path = 'protein_model.pt'
#     best_params_path = model_path.replace('.pt', '_best_params.npy')
#     previous_best_solution = None
#     if os.path.exists(best_params_path):
#         try:
#             previous_best_solution = np.load(best_params_path, allow_pickle=True).item()
#             print("Loaded previous best parameters:", previous_best_solution)
#         except:
#             print("Could not load previous best parameters")
#
#     # Initialize and train HHO classifier with continuation
#     hho_classifier = HHOProteinClassifier(
#         input_shapes,
#         num_hawks=30,
#         num_iterations=10,  # Adjust number of additional iterations as needed
#         model_path=model_path,
#         continue_training=True,
#         previous_best_solution=previous_best_solution
#     )
#
#     model = hho_classifier.train(X_train, y_train, X_test, y_test)
#
#     # Evaluate model
#     evaluate_model(model, X_test, y_test, hho_classifier.device)
#
#     total_time = time.time() - start_time
#     print(f"\nTotal pipeline execution time: {total_time:.2f} seconds")
#     print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

