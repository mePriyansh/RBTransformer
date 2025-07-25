def push_model_to_hub(
    model,
    repo_id,
    local_dir="rbtransformer",
    commit_message="Upload RBTransformer model",
):
    """
    Saves and pushes the model to Hugging Face Hub.

    Args:
        model (torch.nn.Module): The model to push.
        repo_id (str): Hugging Face repository ID.
        local_dir (str): Local directory to save the model before pushing.
        commit_message (str): Commit message for the upload.
    """
    model.save_pretrained(local_dir)
    model.push_to_hub(repo_id=repo_id, commit_message=commit_message)
    print(f"Model uploaded to https://huggingface.co/{repo_id}")
