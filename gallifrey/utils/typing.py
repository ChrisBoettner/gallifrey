#########################
# Code taken from GPJax
#########################

# Copyright 2023 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import TypeAlias, Union

from jaxtyping import Array as JAXArray
from jaxtyping import Bool, Float, Int
from numpy import ndarray as NumpyArray

Array: TypeAlias = Union[JAXArray, NumpyArray]

ScalarArray: TypeAlias = Float[Array, ""]
ScalarBool: TypeAlias = Union[bool, Bool[Array, ""]]
ScalarInt: TypeAlias = Union[int, Int[Array, ""]]
ScalarFloat: TypeAlias = Union[float, Float[Array, ""]]
