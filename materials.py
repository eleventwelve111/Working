import openmc


def create_materials():
    """
    Create and return materials for the simulation:
    - Concrete for the wall
    - Air
    - Void for outside environment
    - Tissue for phantom detector

    Returns:
        openmc.Materials: Collection of materials for the simulation
    """
    materials = openmc.Materials()

    # Concrete (ANSI/ANS-6.4-2006)
    concrete = openmc.Material(name='Concrete')
    concrete.set_density('g/cm3', 2.3)
    concrete.add_element('H', 0.01, 'wo')
    concrete.add_element('C', 0.001, 'wo')
    concrete.add_element('O', 0.529, 'wo')
    concrete.add_element('Na', 0.016, 'wo')
    concrete.add_element('Mg', 0.002, 'wo')
    concrete.add_element('Al', 0.034, 'wo')
    concrete.add_element('Si', 0.337, 'wo')
    concrete.add_element('K', 0.013, 'wo')
    concrete.add_element('Ca', 0.044, 'wo')
    concrete.add_element('Fe', 0.014, 'wo')
    materials.append(concrete)

    # Air (standard composition)
    air = openmc.Material(name='Air')
    air.set_density('g/cm3', 0.001205)
    air.add_element('N', 0.7553, 'wo')
    air.add_element('O', 0.2318, 'wo')
    air.add_element('Ar', 0.0128, 'wo')
    air.add_element('C', 0.0001, 'wo')
    materials.append(air)

    # Void (for outside environment)
    void = openmc.Material(name='Void')
    void.set_density('g/cm3', 1e-10)
    void.add_element('H', 1.0, 'wo')
    materials.append(void)

    # ICRU tissue (for phantom detector)
    tissue = openmc.Material(name='Tissue')
    tissue.set_density('g/cm3', 1.0)
    tissue.add_element('H', 0.101, 'wo')
    tissue.add_element('C', 0.111, 'wo')
    tissue.add_element('N', 0.026, 'wo')
    tissue.add_element('O', 0.762, 'wo')
    materials.append(tissue)

    return materials 