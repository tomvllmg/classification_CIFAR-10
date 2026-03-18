# Attention  il faut split le train_data en  Define the validation set by splitting the training data into 2 subsets (80% training and 20% validation)
# ==============
# La var patience est ici fixé a 10 dans le code mais on peut la passer en argument si on le rzjoute dans le yaml

# Eval function

def eval_cnn_classifier(model, eval_dataloader):

    # Set the model in evaluation mode
    model.eval()

    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        # initialize the total and correct number of labels to compute the accuracy
        correct = 0
        total = 0
        for images, labels in eval_dataloader:
            y_predicted = model(images)
            _, label_predicted = torch.max(y_predicted.data, 1)
            total += labels.size(0)
            correct += (label_predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

# Training with vlidation

def train_val_classifier(model, train_dataloader, valid_dataloader, num_epochs, loss_fn, learning_rate, verbose=True):

    # Make a copy of the model (avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)

    # Set the model in 'training' mode (ensures all parameters' gradients are computed - it's like setting 'requires_grad=True' for all parameters)
    model_tr.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)

    # Initialize a list for storing the training loss over epochs
    train_losses = []

    # EARLY STOPPING
    best_acc = 0
    best_model = None
    list_acc =[]
    
    patience = 10  # Nombre d'epoch à attendre sans amélioration =====> ON PEUT LE PASSER EN PARA SI Jamais
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(num_epochs):

        # Initialize the training loss for the current epoch
        tr_loss = 0

        # Iterate over batches using the dataloader
        for batch_index, (images, labels) in enumerate(train_dataloader):

            # Pareil ici, on ne reshape pas
            pred_labels = model_tr(images)
            loss = loss_fn(pred_labels, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the current epoch loss
            # Note that 'loss.item()' is the loss averaged over the batch, so multiply it with the current batch size to get the total batch loss
            tr_loss += loss.item() * images.shape[0]

        # At the end of each epoch, get the average training loss and store it
        tr_loss = tr_loss/len(train_dataloader.dataset)
        train_losses.append(tr_loss)

        # Display the training loss
        if verbose:
            print('Epoch [{}/{}], Training loss: {:.4f}'.format(epoch+1, num_epochs, tr_loss))

        # LAB 4.1
        accuracy = eval_cnn_classifier(model_tr, valid_dataloader)
        list_acc.append(accuracy)

        # Sauvegarder le meilleur modèle
        if accuracy > best_acc:
          best_acc = accuracy
          best_model = copy.deepcopy(model_tr)
          epochs_without_improvement = 0  # On a progressé, on remet le compteur à zéro
        else:
            epochs_without_improvement += 1 # Pas de progrès, on incrémente

        # CONDITION D'ARRÊT PRÉCOCE
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping déclenché à l'époque {epoch+1}")
            break  # On sort de la boucle 'for epoch'

    torch.save(best_model, 'model_mlp_classif_val_train.pt')

    return best_model, train_losses, list_acc
