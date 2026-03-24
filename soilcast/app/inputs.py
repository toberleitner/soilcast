from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class UserInput:

    till: Literal['conv', 'mintill'] = 'mintill'
    irr: Literal['rf', 'irr'] = 'rf'
    cls: Literal[1, 2, 3, 4, 5, 6, 7] = 1
    rsd: int = 0
    ftn: float | None = None
    location: tuple[float, float] | None = None
    ssp: Literal['126', '585'] = '126'

cls_options = {
    1: 'Productive - cereal', 
    2: 'Productive - wide-row dominated',
    3: 'Productive - mixed',
    4: 'Integrated - cereal',
    5: 'Integrated - wide-row dominated',
    6: 'Integrated - legume dominated',
    7: 'Integrated - mixed'
}

till_options = {
    'conv': 'Conventional',
    'mintill': 'Min/no tillage'
}

irr_options = {
    'rf': 'Rainfed',
    'irr': 'Irrigated'
}

ssp_options = {
    '126': 'SSP 126',
    '585': 'SSP 585'
}