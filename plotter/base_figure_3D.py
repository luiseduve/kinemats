import numpy as np
import plotly
import plotly.graph_objects as go

class PlotterBase():
    """
    Generates an interactive html webpage to plot 3D rotations in the unit circle
    """ 

    # Main Figure setup
    fig = None  # Main figure
    data_objects = []
    layout_setup = None

    # Sphere setup
    mesh_detail = 10
    sphere_radius = 1

    # Arrow setup
    arrow_object = None
    arrow_id = "arrow"
    arrow_color = "black"
    
    def __init__(self, mesh_detail = 10, sphere_radius = 1):
        self.mesh_detail = mesh_detail
        self.sphere_radius = sphere_radius
        pass

    ## Create layout
    def create_layout(self):
        """
        Function to create the layout of the figure
        """
        return go.Layout(title='Sphere Contour Line', 
                        autosize=False, width=700, height=500,
                        margin=dict(l=65, r=50, b=65, t=90),
                        scene = dict(
                            xaxis = dict(
                                #backgroundcolor="rgb(200, 200, 230)",
                                gridcolor="lavender",
                                showbackground=False,
                                zerolinecolor="red",),
                            yaxis = dict(
                                #backgroundcolor="rgb(230, 200,230)",
                                gridcolor="white",
                                showbackground=False,
                                zerolinecolor="green"),
                            zaxis = dict(
                                #backgroundcolor="rgb(230, 230,200)",
                                gridcolor="white",
                                showbackground=False,
                                zerolinecolor="blue",)
                            ))

    ## Create objects
    def create_sphere_vertices(self, step = 10, radius = 1):
        """
        Vertex of a surface that generates a circle

        :param step: Number of subsamples per axis
        :type step: int
        :param radius: Radius of the sphere
        :type step: float
        :return: Cartesian coordinates for the sphere vertices [x y z]
        :rtype: list

        Example:
            x, y, z = create_circle(step = 30, radius = 1)
        """ 
        # Radius equals 1, hence formulas do not consider rho
        theta = np.linspace(0,2*np.pi,step)         # azimuthal angle [0-2pi] in the X-Y plane
        phi = np.linspace(0,np.pi,step)             # polar angle [0-pi] in the Z-axis
        x = radius*np.outer(np.cos(theta),np.sin(phi))
        y = radius*np.outer(np.sin(theta),np.sin(phi))
        z = radius*np.outer(np.ones(step),np.cos(phi))     # radial distance r = 1
        return x, y, z

    def create_sphere(self):
        x, y, z = self.create_sphere_vertices(self.mesh_detail, self.sphere_radius)
        return go.Surface(x=x, y=y, z=z,
                            opacity=0.5,
                            showscale=False,
                            colorscale=plotly.colors.sequential.Blues, # https://plot.ly/python/builtin-colorscales/
                            hoverinfo="none"
                            )
    
    def create_axes_lines(self,axis_length = 0.7):
        axisX= go.Scatter3d(x = [0,axis_length], y = [0,0], z = [0,0], marker = dict( size = .1), line = dict( color = "red", width = 3), showlegend=False)
        axisY= go.Scatter3d(x = [0,0], y = [0,axis_length], z = [0,0], marker = dict( size = .1), line = dict( color = "green", width = 3), showlegend=False)
        axisZ= go.Scatter3d(x = [0,0], y = [0,0], z = [0,axis_length], marker = dict( size = .1), line = dict( color = "blue", width = 3), showlegend=False)

        coneX = go.Cone(x=[axis_length], y=[0], z=[0], u = [1], v = [0], w = [0], sizemode="scaled", sizeref= .2, anchor="cm", colorscale=["red","red"], showscale = False)
        coneY = go.Cone(x=[0], y=[axis_length], z=[0], u = [0], v = [1], w = [0], sizemode="scaled", sizeref=.2, anchor="cm", colorscale=["green","green"], showscale = False)
        coneZ = go.Cone(x=[0], y=[0], z=[axis_length], u = [0], v = [0], w = [1], sizemode="scaled", sizeref=.2, anchor="cm", colorscale=["blue","blue"], showscale = False)

        return [axisX, axisY, axisZ, coneX, coneY, coneZ]
        
    def create_figure(self):
        """
        Creates the figure using plotly library
        :return: Figure setup with a 3D sphere 
        :rtype: plotly.graph_objects.Figure
        """
        self.data_objects = self.create_axes_lines()
        self.data_objects.extend([self.create_sphere()])

        self.layout_setup = self.create_layout()

        self.fig = go.Figure(data = self.data_objects, layout = self.layout_setup)

        return self.fig


    # Create vector indicating the quaternion
    def create_vector_arrow(self, x = 1, y = 1, z=1, id="arrow", color = "black"):
        """
        Creates the plotly objects with the shape of an arrow
        :return:Returns an object with an arrow from the origin to the indicated point
        :rtype: list[go.Scatter3d, go.Cone]
        """
        self.arrow_id = id
        self.arrow_color = color
        self.arrow_object = [go.Scatter3d( x = [0,x], y = [0,y], z = [0,z],
                                marker = dict( size = 2, color = color),
                                line = dict( color = color, width = 6),
                                showlegend=False,
                                idssrc = id),
                            go.Cone(x=[x], y=[y], z=[z], 
                            u = [x], v = [y], w = [z], 
                            sizemode="absolute", sizeref= .2, anchor="cm", 
                            colorscale=[color,color], showscale = False, 
                            idssrc = id+"_cone")]
        return self.arrow_object

    def update_arrow(self, array):
        """
        Updates the position of the arrow trace in the plotly figure
        """
        if(type(array)==type(list()) and len(array)==3):
            x=array[0]
            y=array[1]
            z=array[2]
        else:
            raise ValueError

        position_scatter_x = [0,x]
        position_scatter_y = [0,y]
        position_scatter_z = [0,z]

        # Update lines
        self.fig.update_traces( x=position_scatter_x, y=position_scatter_y, z=position_scatter_z,
                    selector = dict(idssrc = self.arrow_id))

        # Update cone
        self.fig.update_traces( x=[x], y=[y], z=[z], u = [x], v = [y], w = [z], 
                    selector = dict(idssrc = self.arrow_id+"_cone"))

        return self.fig

"""
# Exercise trying to create the mesh3d of the unit sphere. Coordinates of the thriangles where created properly,
# but the variables i,j,k required by mesh3d to generate the triangles has not been done. Surface is going to be
# used instead, although the wireframe cannot be shown.

xv = np.cos(theta) * np.sin(phi)
yv = np.sin(theta) * np.sin(phi)
zv = np.cos(phi)

x, y, z = np.meshgrid(xv, yv, zv)
x = x.reshape(1,-1)[0]
y = y.reshape(1,-1)[0]
z = z.reshape(1,-1)[0]

print(x)
print(y)
print(z)

print("====")
print(np.array(np.meshgrid([1, 2], [4, 5], [6, 7])).T.reshape(-1,3))


sphere = go.Mesh3d(x=x, y=y, z=z,
                    opacity=0.5,
                    alphahull=0,        ### This parameter needs to be changed and i,j,k provided instead
                    showscale=False,
                    colorscale=plotly.colors.sequential.Blues # https://plot.ly/python/builtin-colorscales/
                    
                    )

data = [sphere]  

layout = go.Layout(title='Sphere Contour Line', 
                  autosize=False, width=700, height=500,
                  margin=dict(l=65, r=50, b=65, t=90),
                  scene = dict(
                    xaxis = dict(
                         #backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="lavender",
                         showbackground=False,
                         zerolinecolor="red",),
                    yaxis = dict(
                        #backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="green"),
                    zaxis = dict(
                        #backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="blue",)
                    ))


fig = go.Figure(data = data, layout = layout)
#fig.write_html(PLOTS_FOLDER+'sphere.html', auto_open=True)
fig.show()
"""