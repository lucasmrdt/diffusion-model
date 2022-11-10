# def disp_true_img_grid():
#   for t in range(10):
#     fig, axes = plt.subplots(10, 10, figsize=(10*FIG_SIZE, 10*FIG_SIZE))
#     for i in range(10):
#       for j in range(10):
#         img = axes[i,j].imshow(X_train[y_train == t][i * 10 + j], cmap="gray")
#     plt.show()

# def gen_img_grid():
#   for t in range(10):
#     fig, axes = plt.subplots(10, 10, figsize=(10*FIG_SIZE, 10*FIG_SIZE))
#     for i in range(10):
#       for j in range(10):
#         label = torch.eye(10)[t].to(device)
#         with torch.no_grad():
#           mean_pred = bkw.backward_loop(label[None], (1,28,28), progress_bar=False)[-1]
#         mean_pred = mean_pred.detach().cpu().squeeze(0)
#         axes[i,j].imshow(mean_pred, cmap="gray")
#     plt.show()