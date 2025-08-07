import numpy as np
import vtk


def create_structured_grid(points, nx, ny, nz):
    '''
    Creates a structured grid for the given points.
    '''
    # https://examples.vtk.org/site/Cxx/StructuredGrid/StructuredGrid/
    points_vtk = vtk.vtkPoints()
    for i in range(points.shape[1]):
        if points.shape[0] == 2:
            points_vtk.InsertNextPoint(points[0, i], points[1, i], 0)
        elif points.shape[0] == 3:
            points_vtk.InsertNextPoint(points[0, i], points[1, i], points[2, i])
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, nz)
    grid.SetPoints(points_vtk)
    return grid


def add_cell_array(mesh, values, name):
    '''
    Adds the given cell values to the given mesh.
    '''
    # https://examples.vtk.org/site/Cxx/PolyData/Casting/
    array = vtk.vtkDoubleArray()
    if len(values.shape) == 1:
        array.SetNumberOfComponents(1)
    else:
        array.SetNumberOfComponents(values.shape[1])
    array.SetName(name)
    for e in range(mesh.GetNumberOfCells()):
        array.InsertNextValue(values[e])
    mesh.GetCellData().AddArray(array)


def add_point_array(mesh, values, name):
    '''
    Adds a point array to the given mesh.
    '''
    # https://examples.vtk.org/site/Cxx/PolyData/Casting/
    array = vtk.vtkDoubleArray()
    if len(values.shape) == 1:
        array.SetNumberOfComponents(1)
    else:
        array.SetNumberOfComponents(values.shape[1])
    array.SetName(name)
    for i in range(mesh.GetNumberOfPoints()):
        array.InsertNextValue(values[i])
    mesh.GetPointData().AddArray(array)


def write_structured_grid(mesh, vts_file_path):
    '''
    Writes the given mesh to the given vts file path.
    '''
    # https://examples.vtk.org/site/Cxx/IO/XMLStructuredGridWriter/
    print(f'Writing {vts_file_path}')
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(vts_file_path)
    writer.SetInputData(mesh)
    writer.Write()


def create_multi_block(meshes, names=None):
    '''
    Creates a MultiBlockDataSet with the given meshes.
    '''
    multi_block = vtk.vtkMultiBlockDataSet()
    for i in range(len(meshes)):
        multi_block.SetBlock(i, meshes[i])
        if names is not None:
            info = multi_block.GetMetaData(i)
            info.Set(vtk.vtkCompositeDataSet.NAME(), names[i])
    return multi_block


def write_multi_block(multi_block, vtm_file_path):
    '''
    Writes the given MultiBlockDataSet to the given vtm file path.
    '''
    print(f'Writing {vtm_file_path}')
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetFileName(vtm_file_path)
    writer.SetInputData(multi_block)
    writer.Write()


def read_vts_file(vts_file_path):
    '''
    Reads the given vts file.
    '''
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(vts_file_path)
    reader.Update()
    result = reader.GetOutput()
    return result


def read_phi_from_vts_file(vts_file_path):
    '''
    Reads the phi array from the given vts file.
    '''
    result = read_vts_file(vts_file_path)
    n_points = result.GetNumberOfPoints()
    point_data = result.GetPointData()
    n_arrays = point_data.GetNumberOfArrays()
    array_names = [point_data.GetArrayName(i) for i in range(n_arrays)]
    phi_index = [i for i in range(n_arrays) if array_names[i] == 'phi'][0]
    phi_array = point_data.GetArray(phi_index)
    phi = np.array([phi_array.GetValue(i) for i in range(n_points)])
    return phi

