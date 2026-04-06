import torch
import copy
import wandb

def eval_cnn_classifier(model, eval_dataloader):

    # On passe en mode évaluation
    model.eval()

    # Dans la phase de test on ne calcule pas les gradients 
    with torch.no_grad():
        # Initialise le nombre total et correct de labels pour calculer la précision
        correct = 0
        total = 0
        for images, labels in eval_dataloader:
            y_predicted = model(images)
            _, label_predicted = torch.max(y_predicted.data, 1)
            total += labels.size(0)
            correct += (label_predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy
    

def train_val_classifier(model_tr, train_dataloader, valid_dataloader, num_epochs, loss_fn, optimizer, scheduler, patience, verbose=True):

    # Liste pour stocker les pertes au cours de epochs
    train_losses = []

    # Early stopping
    best_acc = 0
    best_model = None
    list_acc =[]
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(num_epochs):
        
        model_tr.train()
        # Initialise la perte
        tr_loss = 0

        # Iteration les batches
        for batch_index, (images, labels) in enumerate(train_dataloader):
            pred_labels = model_tr(images)
            loss = loss_fn(pred_labels, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * images.shape[0]

        # A la fin de chaque epoch, on stock la perte moyenne d'entrainement
        tr_loss = tr_loss/len(train_dataloader.dataset)
        train_losses.append(tr_loss)

        # Affichage
        if verbose:
            print('Epoch [{}/{}], Training loss: {:.4f}'.format(epoch+1, num_epochs, tr_loss))

        accuracy = eval_cnn_classifier(model_tr, valid_dataloader)
        list_acc.append(accuracy)

        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_accuracy": accuracy,
            "learning_rate": optimizer.param_groups[0]['lr'] 
        })

        # Sauvegarder le meilleur modèle
        if accuracy > best_acc:
          best_acc = accuracy
          best_model = copy.deepcopy(model_tr)
          epochs_without_improvement = 0  # On a progressé, on remet le compteur à zéro
        else:
            epochs_without_improvement += 1 # Pas de progrès, on incrémente

        # Condition d'arret précoce
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping déclenché à l'époque {epoch+1}")
            break  # On sort de la boucle 'for epoch'
            
        if scheduler is not None:
            scheduler.step()

    torch.save(best_model, 'model_classif_val_train.pt')

    return best_model, train_losses, list_acc
