from PIL import Image
import clip  # pylint: disable=import-outside-toplevel
import webdataset as wds  # pylint: disable=import-outside-toplevel
import torch
import torchvision
import torchvision.transforms as T
import io
import os

class MyWebDataset(wds.WebDataset):
    def __init__(
            self,
            urls, **kwargs
    ):
        super().__init__(urls, **kwargs)

    def iterator(self):
        """Create an iterator through the entire dataset, using the given number of repetitions."""
        for i in range(self.repetitions):
            for sample in self.iterator1():
                if sample['image'].mean() <= -0.95:
                    print(sample['image_filename'])
                    continue
                else:
                    yield sample

def create_webdataset(
    data_root,
    image_size,
    batch_size,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    # import clip  # pylint: disable=import-outside-toplevel
    # import webdataset as wds  # pylint: disable=import-outside-toplevel

    urls = os.listdir(data_root)
    urls = [url for url in urls if url.endswith(".tar")]
    urls = [os.path.join(data_root, url) for url in urls]
    #dataset = wds.ShardList(urls, splitter=wds.split_by_worker, nodesplitter=wds.split_by_node, shuffle=False)
    #dataset = wds.Processor(dataset, wds.url_opener)
    #dataset = wds.Processor(dataset, wds.tar_file_expander)
    #dataset = wds.Processor(dataset, wds.group_by_keys)
    #dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue)#.with_epoch(10000)
    #dataset = wds.WebDataset(urls).decode("pil").batched(batch_size, partial=False)
    # dataset = wds.WebDataset(urls, resampled=True)
    dataset = MyWebDataset(urls, resampled=True)
    tokenizer = lambda text: clip.tokenize([text], truncate=True)[0]
    image_transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        output = {}
        if enable_image:
            image_data = item[image_key]
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_tensor = image_transform(image)
            output["image_filename"] = item["__key__"]
            output["image"] = image_tensor * 2 - 1.

        if enable_text:
            text = item[caption_key]
            caption = text.decode("utf-8")
            tokenized_text = tokenizer(caption)
            output["cond"] = tokenized_text
            output["text"] = caption

        if enable_metadata:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            output["metadata"] = metadata
        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue).with_epoch(10000)
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, shuffle=False):
    """Create a pytorch dataloader from a dataset"""

    # def collate_fn(batch):
    #     batch = list(filter(lambda x: x is not None, batch))
    #     return default_collate(batch)

    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        # collate_fn=collate_fn if input_format == "files" else None,
    )
    return data


class WebdatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        data_root,
        image_size,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="jpg",
        wds_caption_key="txt",
        cache_path=None, **kwargs
    ):
        self.batch_size = batch_size
        dataset = create_webdataset(
            data_root,
            image_size,
            batch_size,
            enable_text=enable_text,
            enable_image=enable_image,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            cache_path=cache_path,
        )
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers)
        #loader = wds.WebLoader(dataset, num_workers=num_prepro_workers)
        #self.dataloader = loader.ddp_equalize(1000000 // batch_size)

    def __iter__(self):
        for batch in self.dataloader:
            yield batch

if __name__ == '__main__':
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tar_folder = "/home/huang/Downloads/mobaxterm/webdata"
    # input_dataset = [tar_folder + "/00807.tar", tar_folder + "/00000.tar", ]
    input_dataset = [tar_folder + "/00807.tar"]
    batch_size = 2
    num_prepro_workers = 2
    device = torch.device('cpu')
    model, _ = clip.load("ViT-B/32", device='cpu')

    preprocess = torchvision.transforms.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    output_partition_count = 2
    actual_values = []
    data_loader = WebdatasetReader(
        data_root=tar_folder,
        image_size=(224, 224),
        batch_size=batch_size,
        num_prepro_workers=num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=True,
    )

    for d in data_loader:
        text_features = model.encode_text(d['cond'])
        image_features = model.encode_image(d['image'])

        logits_per_image, logits_per_text = model(d['image'], d['cond'])
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        continue