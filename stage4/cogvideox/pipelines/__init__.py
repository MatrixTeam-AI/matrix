from typing import TYPE_CHECKING

from diffusers.utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_cogvideox"] = ["CogVideoXPipeline", "CogVideoXStreamingPipeline", "CogVideoXStreamingPipelinePCFG"]
    _import_structure["pipeline_cogvideox_fun_control"] = ["CogVideoXFunControlPipeline"]
    _import_structure["pipeline_cogvideox_image2video"] = ["CogVideoXImageToVideoPipeline"]
    _import_structure["pipeline_cogvideox_video2video"] = ["CogVideoXVideoToVideoPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from diffusers.utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_cogvideox import CogVideoXPipeline, CogVideoXStreamingPipeline, CogVideoXStreamingPipelinePCFG
        from .pipeline_cogvideox_fun_control import CogVideoXFunControlPipeline
        from .pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
        from .pipeline_cogvideox_video2video import CogVideoXVideoToVideoPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
