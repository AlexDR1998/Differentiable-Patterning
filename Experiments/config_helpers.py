

def build_tags(cfg, prefix=""):
    tags = []
    for key, value in cfg.items():
        if key == "seed":
            continue
        tag_key = f"{prefix}{key}"
        if value is None:
            continue
        if hasattr(value, "items"):
            tags.extend(build_tags(value, prefix=f"{tag_key}."))
        else:
            tags.append(f"{tag_key}:{value}")
    return tags