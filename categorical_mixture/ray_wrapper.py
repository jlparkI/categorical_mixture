"""A simple wrapper for the C extension used if the Ray
package is used for multiprocessing. This code is separate
from the MultinoulliMixture so that users can run
the model without Ray installed if they do not want to
use it."""
import ray
from core_cpu_func_wrappers import em_offline

@ray.remote
def _ray_caller(xchunk, mix_weights, mu_mix):
    """A simple wrapper for the C extension that ray can use."""
    res = em_offline(xchunk, mix_weights.copy(), mu_mix.copy(), 1)
    return res
