# ======================================================================================================
# Comme on veut de l'early stopping ce fichier ests inutile : on fait l'early stopping dan sl'autre fichier 
# ======================================================================================================


def training_classifier(model, train_dataloader, num_epochs, loss_fn, optimizer, verbose=True):

    # Make a copy of the model (avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)

    # Set the model in 'training' mode (ensures all parameters' gradients are computed - it's like setting 'requires_grad=True' for all parameters)
    model_tr.train()

    # The optimizer is defined in the build optimizer

    # Initialize a list for storing the training loss over epochs
    train_losses = []

    # Training loop
    for epoch in range(num_epochs):

        # Initialize the training loss for the current epoch
        tr_loss = 0

        # Iterate over batches using the dataloader
        for batch_index, (images, labels) in enumerate(train_dataloader):

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

    return model_tr, train_losses


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
