from models.ltx.ltx_adapter import LTXAdapter
from models.stable_video.svd_adapter import SVDAdapter
from models.custom_model.custom_adapter import CustomAdapter

ADAPTERS = {
    "ltx2_22B": LTXAdapter,
    "ltx2_22B_distilled": LTXAdapter,
    "svd": SVDAdapter,
    "custom": CustomAdapter,
}
