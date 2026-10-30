# The rerank extra installs torch; the dev environment does not. This stub
# covers only what vecgrep calls so the type gate does not need CUDA wheels.
from . import cuda as cuda

class dtype: ...

float16: dtype

def set_num_threads(num: int, /) -> None: ...
