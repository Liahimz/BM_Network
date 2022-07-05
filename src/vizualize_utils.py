import plotly
# import kaleido
import plotly.express as px

def PlotScatter(X, y, plot_name):
    fig = px.scatter(None, x=X[:,0], y=X[:,1], 
                     labels={
                         "x": "Dimension 1",
                         "y": "Dimension 2",
                     },
                     opacity=1, color=y)

    fig.update_layout(dict(plot_bgcolor = 'white'))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_layout(title_text=plot_name)

    fig.update_traces(marker=dict(size=8,
                                 line=dict(color='black', width=0.3)))
    
    fig.write_image("pics/" + plot_name +  ".png")
    plotly.offline.plot(fig, filename="html/" + plot_name + '.html')
    return fig

def PlotLine(X, y, plot_name):
    fig = px.line(None, x=X, y=y, 
                     labels={
                         "x": "Dimension 1",
                         "y": "Dimension 2",
                     },
                    )

    fig.update_layout(dict(plot_bgcolor = 'white'))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_layout(title_text=plot_name)

    fig.update_traces(marker=dict(size=8,
                                 line=dict(color='black', width=0.3)))
    
    fig.write_image("pics/" + plot_name +  ".png")
    plotly.offline.plot(fig, filename="html/" + plot_name + '.html')
    return fig

def Plot3D(X, y, plot_name):
    fig = px.scatter_3d(None, 
                        x=X[:,0], y=X[:,1], z=X[:,2],
                        color=y,
                        height=1000, width=1000
                       )
    fig.update_layout(title_text=plot_name,
                      showlegend=False,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                            center=dict(x=0, y=0, z=-0.1),
                                            eye=dict(x=1.5, y=1.75, z=1)),
                                            margin=dict(l=0, r=0, b=0, t=0),
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             ),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                              ),
                                   zaxis=dict(backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             )))
    fig.update_traces(marker=dict(size=4, 
                                  line=dict(color='black', width=0.1)))
    fig.update(layout_coloraxis_showscale=False)
    fig.write_image(plot_name +  ".png")
    return fig