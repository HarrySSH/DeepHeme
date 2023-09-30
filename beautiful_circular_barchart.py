# Import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_circular_barchart(number_list, cell_types, labelPadding=1):
    # initialize the figure
    plt.figure(figsize=(20,10))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')


    lowerLimit = 5
    # Compute max and min in the dataset
    _max = max(number_list)
    _min = min(number_list)

    import math

    slope = (_max - _min) / _max
    heights = [math.log2(slope * x +3)  for x in number_list]

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(cell_types)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(cell_types)+1))
    angles = [element * width for element in indexes] 
    COLORS = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#1f1f1f",
            "#ff0000",
            '#d3d3d3',
        ]
    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2, 
        # color is as it is if the number is positive, otherwise it is gray
        color=[COLORS[i] if number_list[i] > 0 else "#d3d3d3" for i in range(len(cell_types))],
        edgecolor="white")


    # Add labels
    for bar, angle, height, label, number in zip(bars,angles, heights, cell_types, number_list):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180 
        else: 
            alignment = "left" 
            rotation = rotation 

        # Finally add the labels, the angle should be vertical  to the bar:
        rotation = rotation + 90
        ax.text(

            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding , 
            s=label, #+'\n'+str(number), 
            ha=alignment, 
            va='top', 
            rotation=rotation, 
            ### make the label at the center of the bar

            rotation_mode="anchor",
            weight="bold",
        size = 20) 
        # number is bold 
        ax.text(
            x=angle, 
            y=lowerLimit + min(heights), 
            s=str(number), 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor",
            weight="bold",
        size = 20) 
    plt.tight_layout()
    #plt.show()
        
    plt.savefig('beautiful_circular_barchart.png')

def piechart(number_list, cell_types, centre_circle=0.5, line_guide=False):
    
    # Pie chart, where the slices will be ordered and plotted counter-clockwise
    # the label will be similar to the function above
    labels = cell_types
    sizes = number_list
    explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05,0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    # put a hole in the middle
    
    
    COLORS = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#1f1f1f",
            "#ff0000",
            '#d3d3d3',
        ]
    # add another not important color for the other cells
    
    
    
    
    if not line_guide:
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90, colors=COLORS)
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()
        
        plt.show()
    else:
        bbox_props=dict(boxstyle='square,pad=0.3',fc ='w',ec='k',lw=0.92)
        kw=dict(xycoords='data',textcoords='data',arrowprops=dict(arrowstyle='-'),zorder=0,va='center')

        fig1,ax1=plt.subplots(figsize=(20,13))
        
        values=number_list
        labels = [_celltype+'\n' + str(_value) + '%' for _celltype, _value in zip(cell_types, values)]
        # Add code
        annotate_dict = zip(labels, values)
        annotate_dict_cp = zip(labels, values)
        
        # color as well
        COLORS_dict =  zip(COLORS, values)
        COLORS_dict_cp =  zip(COLORS, values)
        
        val = [[x,y] for x,y in zip(sorted(values, reverse=True),sorted(values))]
        annotated_list = []
        for x,y in zip(sorted(annotate_dict, key=lambda x: x[1], reverse=True),  
                                               sorted(annotate_dict_cp, key=lambda x: x[1])):
            
            annotated_list.append([x,y])
            

        color_list = []
        for x,y in zip(sorted(COLORS_dict, key=lambda x: x[1], reverse=True),  
                                               sorted(COLORS_dict_cp, key=lambda x: x[1])):
            
            color_list.append([x,y])
            
        
            
        assert len(annotated_list) !=0, 'annotated_list is empty'


        # conver the color as well
        
        values1 = sum(val, [])
        annotate_dict = sum(annotated_list, [])
        color_dict = sum(color_list, [])
        
        new_labels = []
        new_colors = [] 
        
        for i in range(len(values)):
            
            new_labels.append(annotate_dict[i][0])
            #print(color_list[i])
            #print(color_list[i][0])
            new_colors.append(color_dict[i][0])
        #sdvs
        
                    
        wedges,texts=ax1.pie(values1[:len(values)],labeldistance=1.2,startangle=90,
                             colors=new_colors,
                             
                               explode=explode)
        
        for i,p in enumerate(wedges):
            if 'Other' in new_labels[i]:
                continue
            ang=(p.theta2-p.theta1)/2. +p.theta1
            y=np.sin(np.deg2rad(ang))
            x=np.cos(np.deg2rad(ang))
             
            horizontalalignment={-1:"right",1:"left"}[int(np.sign(x))]
            
            connectionstyle="angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle":connectionstyle})
            ax1.annotate(new_labels[i],xy=(x, y),xytext=(1.35*np.sign(x),1.4*y),
                        horizontalalignment=horizontalalignment,**kw, fontsize=30,
                        weight='bold')
        
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
            
        
        plt.savefig('piechart.png')
    

        


cell_types =['Myeloblast', 'Promyelocyte', 'Myelocyte','Metamyelocyte',
                      "Band\nneutrophil", "Segmented\nneutrophil",
                      'Eosinophil', 
                     'Basophil', 'Monocyte', 'Lymphocyte', 'Plasma cell', 'Normoblast', 'Others']

labelPadding = 1
number_list = [9, 0,0,1,2,1, 1, 0, 1, 4, 0, 2,1]
plot_circular_barchart(number_list, cell_types)
# 12 number sum to 100
number_list = [47,2,3,2, 7, 7, 3, 1, 4, 15, 2,6, 1]



piechart(number_list, cell_types, centre_circle=0.8, line_guide=True)


