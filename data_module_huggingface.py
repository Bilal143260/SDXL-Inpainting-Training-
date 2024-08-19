from datasets import load_dataset
import torch
from torchvision import transforms
import random
import numpy as np

#first set export HF_TOKEN=""

#function copied from:
#https://github.com/huggingface/diffusers/blob/main/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py
def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    # if masks are in binary, comment the below two lines
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def make_train_dataset(tokenizer1, tokenizer2):
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = load_dataset("BoooomNing/SAM_BG")

    # Preprocessing the datasets.
    column_names = dataset["train"].column_names

    #target_images
    image_column = column_names[0]

    #prompts
    caption_column = column_names[4]
        
    #mask images
    conditioning_image_column = column_names[2]

    def tokenize_captions1(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            # if random.random() < args.proportion_empty_prompts:
            if random.random() < 0.2:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer1(
            captions, max_length=tokenizer1.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def tokenize_captions2(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            # if random.random() < args.proportion_empty_prompts:
            if random.random() < 0.2:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer2(
            captions, max_length=tokenizer2.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def prepare_images(examples):
        masks = []
        masked_images = []

        for target_image, mask_image in zip(examples[image_column],examples[conditioning_image_column]):
            mask,masked_image = prepare_mask_and_masked_image(target_image, mask_image)
            masks.append(mask)
            masked_images.append(masked_image)

        return masks, masked_images

    image_transforms = transforms.Compose(
        [
            # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        examples["target_images"] = images
        examples["masks"],examples["masked_images"] = prepare_images(examples)
        examples["text_input_ids"] = tokenize_captions1(examples)
        examples["text_input_ids_2"] = tokenize_captions2(examples)

        return examples

    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    target_images = torch.stack([example["target_images"] for example in examples])
    target_images = target_images.to(memory_format=torch.contiguous_format).float()

    masks = torch.cat([example["masks"] for example in examples], dim=0)
    masks = masks.to(memory_format=torch.contiguous_format).float()

    masked_images = torch.cat([example["masked_images"] for example in examples], dim=0)
    masked_images = masked_images.to(memory_format=torch.contiguous_format).float()

    text_input_ids = torch.stack([example["text_input_ids"] for example in examples])
    text_input_ids_2 = torch.stack([example["text_input_ids_2"] for example in examples])

    return {
        "target_images": target_images,
        "masks": masks,
        "masked_images": masked_images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2
    }

if __name__ == "__main__":
    from transformers import CLIPTokenizer
    pretrained_model_name_or_path = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
    train_dataset = make_train_dataset(tokenizer,tokenizer_2)

    print(train_dataset[0]["pixel_values"].shape)

    data = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=4,
        num_workers=1)

    # Retrieve the first batch
    first_batch = next(iter(data))

    # Print the shape of "masked_images"
    print(first_batch["pixel_values"].shape)
    





