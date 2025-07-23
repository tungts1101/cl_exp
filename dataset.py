

def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=16, val_fraction=0.1, max_val_samples=5000, subset_config=None):
    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers, subset_config=subset_config
    )
    return dataset


def get_dataset_and_classifier_for_split(dataset, split_idx, text_encoder, args, remap_labels=True, return_classifier=True):
    if args.split_strategy == 'data':
        train_subset_indices, test_subset_indices = \
            get_balanced_data_incremental_subset_indices(
                dataset.train_dataset, args.n_splits, split_idx
            )
        dataset.train_dataset = torch.utils.data.Subset(dataset.train_dataset, train_subset_indices)
        # it does not make sense to split test in data-incremental
        # dataset.test_dataset = torch.utils.data.Subset(dataset.test_dataset, test_subset_indices)
        if return_classifier:
            classification_head = get_classification_head(args, args.dataset)
    elif args.split_strategy == 'class':
        classes, train_subset_indices, test_subset_indices = \
            get_class_incremental_classes_and_subset_indices(
                dataset, args.n_splits, split_idx
            )

        dataset.train_dataset = Subset(dataset.train_dataset, train_subset_indices)
        dataset.test_dataset = Subset(dataset.test_dataset, test_subset_indices)

        if remap_labels:
            class_map = {c: idx for idx, c in enumerate(sorted(classes))}        
            dataset.train_dataset.dataset.target_transform = lambda t : class_map[t]
            dataset.test_dataset.dataset.target_transform = lambda t : class_map[t]

        if return_classifier:
            classification_head = build_subset_classification_head(
                text_encoder.model, args.dataset, classes, args.data_location, args.device
            )
    else:
        raise NotImplementedError()
    
    # dataloaders
    dataset.train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset, batch_size=dataset.train_loader.batch_size,
        shuffle=True, num_workers=dataset.train_loader.num_workers
    )        
    dataset.test_loader = torch.utils.data.DataLoader(
        dataset.test_dataset, batch_size=dataset.test_loader.batch_size,
        shuffle=False, num_workers=dataset.test_loader.num_workers
    )

    return (dataset, classification_head) if return_classifier else dataset