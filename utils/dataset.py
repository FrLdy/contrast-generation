def compute_mean_std(dataloader):
    mean = 0
    std = 0
    nb_samples = 0
    for batch in dataloader:
        batch_size = batch.size(0)
        data = batch.view(batch_size, batch.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_size

    return mean / nb_samples, std / nb_samples