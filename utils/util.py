import torch
import torch.nn as nn
import pandas as pd
import datetime
import os


def training_model(args, model, train_loader, test_loader):

    model.to(args.device)  # 将模型移动到GPU设备上

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    dfhistory = pd.DataFrame(columns=['epoch', 'train_loss', 'train_metric', 'val_loss', 'val_metric'])

    for epoch in range(1, args.num_epochs + 1):
        # 1. Training loop
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(args.device) for k, v in batch.items()}  # 将数据移动到GPU设备上

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy as the metric
            _, predictions = torch.max(logits, dim=1)
            correct = torch.sum(predictions == labels)
            metric = correct.item() / labels.size(0)

            loss_sum += loss.item()
            metric_sum += metric

            log_step_freq = 30
            metric_name = "accuracy"
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                    (step, loss_sum / step, metric_sum / step))

        # 2. Validation loop
        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1
        best_loss = float('inf')

        with torch.no_grad():
            for val_step, batch in enumerate(test_loader, 1):
                batch = {k: v.to(args.device) for k, v in batch.items()}  # 将数据移动到GPU设备上

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                logits = model(input_ids, attention_mask)

                val_loss = loss_fn(logits, labels)

                _, predictions = torch.max(logits, dim=1)
                correct = torch.sum(predictions == labels)
                val_metric = correct.item() / labels.size(0)

                val_loss_sum += val_loss.item()
                val_metric_sum += val_metric

        # 3. Record and print logs
        info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info
        
        if val_loss_sum / val_step < best_loss:
            best_loss = val_loss_sum / val_step
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))

        print(f"\nEPOCH = {epoch}, loss = {loss_sum / step:.3f}, metric = {metric_sum / step:.3f}, " +
            f"val_loss = {val_loss_sum / val_step:.3f}, val_metric = {val_metric_sum / val_step:.3f}")

        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "=" * 16 + f"{nowtime}")

    print('Finished Training...')

    return dfhistory
