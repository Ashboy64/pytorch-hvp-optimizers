import matplotlib.pyplot as plt


def vizualize_results(experiment_name, out_timesteps, out_train_loss, out_val_loss, out_val_acc):
    # plt.figure(figsize=(10, 4), dpi=80)
    plt.plot(out_timesteps, out_val_loss, label=f"{experiment_name} val loss")

    # plt.subplot(1, 2, 1)
    # plt.plot(out_timesteps, out_train_loss, label="train loss")
    # plt.plot(out_timesteps, out_val_loss, label="val loss")
    # plt.xlabel("num training samples")
    # plt.ylabel("cross-entropy loss")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(out_timesteps, out_val_acc, label="val acc")
    # plt.xlabel("num training samples")
    # plt.ylabel("accuracy")
    # plt.legend()

    