import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

class SST5Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                text, label = line.strip().split("\t")
                self.texts.append(text)
                self.labels.append(str(label))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Prepare input and target for T5
        input_text = f"Classify: {text}"
        target_text = label

        encoded_input = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded_target = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=2, # Target為一個字，因此長度為2
            return_tensors="pt",
        )
        # print("Encoded Target:", encoded_target["input_ids"])
        # print("Decoded Target:", tokenizer.decode(encoded_target["input_ids"][0]))


        return {
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "labels": encoded_target["input_ids"].squeeze(0),
        }

class SoftPrompt(nn.Module):
    def __init__(self, prompt_length, embedding_dim):
        super(SoftPrompt, self).__init__()
        self.prompt_embeddings = nn.Embedding(prompt_length, embedding_dim)
        self.prompt_length = prompt_length

    def forward(self, batch_size):
        # Generate prompt embeddings for the batch
        prompt_ids = torch.arange(self.prompt_length, device=self.prompt_embeddings.weight.device)
        prompt_embeddings = self.prompt_embeddings(prompt_ids).unsqueeze(0).expand(batch_size, -1, -1)
        return prompt_embeddings


# Initialize T5 and Soft Prompt
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Freeze T5 model parameters
for param in model.parameters():
    param.requires_grad = False

# Dataset and DataLoader
batch_size = 16
# train_file = "C:\\梁旂\\NSYSU\\VScode\\EvoPrompt\\EvoPrompt\\data\\cls\\sst-5\\seed10\\dev_500.txt"
train_file = "./dev_500.txt"
# test_file = "C:\\梁旂\\NSYSU\\VScode\\EvoPrompt\\EvoPrompt\\data\\cls\\sst-5\\seed10\\test.txt"
test_file = "./test.txt"
train_dataset = SST5Dataset(train_file, tokenizer)
test_dataset = SST5Dataset(test_file, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

### Training ###
random.seed(2024)
best_val_accuracy = -1
best_soft_prompt = None
length_list = [10, 15, 20]

for soft_num in range(3):
    # Initialize Soft Prompt
    # prompt_length = random.randint(10, 20)
    # prompt_length = 10
    prompt_length = length_list[soft_num]
    embedding_dim = model.config.d_model
    soft_prompt = SoftPrompt(prompt_length, embedding_dim)
    print("Soft Prompt Initial Weights:")
    print(soft_prompt.prompt_embeddings.weight)

    # Optimizer
    optimizer = AdamW(soft_prompt.parameters(), lr=0.001)

    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    soft_prompt.to(device)

    # Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Generate soft prompt embeddings
            prompt_embeddings = soft_prompt(batch_size=input_ids.size(0))
            # Combine soft prompt with input embeddings
            input_embeddings = model.get_input_embeddings()(input_ids)
            combined_embeddings = torch.cat([prompt_embeddings, input_embeddings], dim=1)
            # Extend attention mask
            extended_attention_mask = torch.cat(
                [torch.ones((attention_mask.size(0), prompt_length), device=device), attention_mask], dim=1
            )

            # Forward pass
            outputs = model(
                inputs_embeds=combined_embeddings,
                attention_mask=extended_attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            # # Debugging: Print Soft Prompt gradients
            # for name, param in soft_prompt.named_parameters():
            #     if param.requires_grad:
            #         print(f"Parameter: {name}, Grad Norm: {param.grad.norm() if param.grad is not None else 'None'}")

            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
        # # Debugging: Print Soft Prompt Weight Changes
        # if epoch == 0:
        #     initial_weights = soft_prompt.prompt_embeddings.weight.clone()
        # weight_diff = (soft_prompt.prompt_embeddings.weight - initial_weights).abs().mean().item()
        # print(f"Soft Prompt Weight Change After Epoch {epoch + 1}: {weight_diff:.6f}")


    # Val (test_file)
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Generate soft prompt embeddings
            prompt_embeddings = soft_prompt(batch_size=input_ids.size(0))

            # Combine soft prompt with input embeddings
            input_embeddings = model.get_input_embeddings()(input_ids)
            combined_embeddings = torch.cat([prompt_embeddings, input_embeddings], dim=1)

            # Extend attention mask
            extended_attention_mask = torch.cat(
                [torch.ones((attention_mask.size(0), prompt_length), device=device), attention_mask], dim=1
            )

            # Generate outputs
            outputs = model.generate(inputs_embeds=combined_embeddings, attention_mask=extended_attention_mask, max_length=2)
            # predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # targets = [tokenizer.decode(label, skip_special_tokens=True).strip() for label in labels]
            predictions = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
            predictions = [pred if pred != "" else "0" for pred in predictions]         
            targets = [tokenizer.decode(label[label != -100], skip_special_tokens=True).strip() for label in labels]
            targets = [targ if targ != "" else "0" for targ in targets]  # 0不知為何會對應為''，因此將''設為 "0"

            # Debugging: Print predictions and targets
            print("Generated Token IDs:", outputs)
            print(f"Predictions: {predictions}")
            print(f"Targets: {targets}")

            correct_predictions += sum(p == t for p, t in zip(predictions, targets))
            total_predictions += len(targets)

    val_accuracy = correct_predictions / total_predictions
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Save the best soft prompt
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        # best_soft_prompt = soft_prompt.prompt_embeddings.weight.detach().cpu()
        best_soft_prompt_state = soft_prompt.state_dict()
        print(f"Best soft prompt updated: Soft Prompt {soft_num}")

# Save best soft prompt
if best_soft_prompt_state is not None:
    torch.save(best_soft_prompt_state, "best_soft_prompt_.pt")
    print("Best soft prompt saved.")
else:
    print("Best soft prompt not found.")