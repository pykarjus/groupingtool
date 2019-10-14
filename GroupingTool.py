import base64
import io

import json
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
from plotly.graph_objs import *
import dash_table as dt
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colorscale=[[0.0, "rgb(165,0,38)"],
            [0.1111111111111111, "rgb(215,48,39)"],
            [0.2222222222222222, "rgb(244,109,67)"],
            [0.3333333333333333, "rgb(253,174,97)"],
            [0.4444444444444444, "rgb(254,224,144)"],
            [0.5555555555555556, "rgb(224,243,248)"],
            [0.6666666666666666, "rgb(171,217,233)"],
            [0.7777777777777778, "rgb(116,173,209)"],
            [0.8888888888888888, "rgb(69,117,180)"],
            [1.0, "rgb(49,54,149)"]]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Group Tool'
server = app.server
app.config['suppress_callback_exceptions'] = True

layout = html.Div(
    [
        dbc.Row([dbc.Col(html.Div([dcc.Upload(id='similaritymatrix',children=html.Div(['Drag and Drop "SimilarityMatrix" or ', html.A('Select Files')]),
                                              style={'width': '100%', 'height': '60px','lineHeight': '60px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px'}, multiple=False),
                                   html.Div(id='similaritymatrixoutput', style={'display': 'none'})
                                  ]
                                 )
                        ),
                 dbc.Col(html.Div([dcc.Upload(id='taglist',children=html.Div(['Drag and Drop "Taglist" or ', html.A('Select Files')]),
                                              style={'width': '100%', 'height': '60px','lineHeight': '60px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px'}, multiple=False),
                                   html.Div(id='taglistoutput', style={'display': 'none'})
                                  ]
                                 )
                        )]),
        dbc.Row(
            [
                dbc.Col(html.Div(id='output', style={'display': 'none'})),
                dbc.Col(html.Div(id='output2', style={'display': 'none', 'height': 500})),
                dbc.Col(html.Div(id='testdiv', style={'display': 'none'})),
                dbc.Col(html.Div(id='testdiv2', style={'display': 'none'}))
            ]),
        dbc.Row([dbc.Col([dbc.Row(html.Div(id='tablediv')),
                         dbc.Row(html.Div(id='tablediv2'))]),
                dbc.Col(html.Div(id='groupkpis'), width = 3)]),
        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id='group dropdown',options=[],placeholder = "SELECT GROUP"))
            ]),
        dbc.Row(
            [
                dbc.Col(dbc.Row([html.Div(id='graafin title'), html.Div(id ='graafi', style={'width': '100%'})], justify="center")),
                dbc.Col(dbc.Row([html.Div(id='tags for group title'),html.Div(id = 'tags for group', style={'width': '100%'})], justify="center"), width = 6)
            ]),
        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id='tag dropdown',options=[],placeholder = "SELECT TAG")),
            ]),
        dbc.Row(
            [
                dbc.Col(dbc.Row([html.Div(id='heatmaptitle'),html.Div(id='tagheatmap')], justify = 'center')),
                dbc.Col(dbc.Row([html.Div(id='tagsimmatrix')], justify = 'start')),
            ])
    ]
)

app.layout = layout

def ModifyInputFiles(SimFile,GroupsFile):
    SimFileInd=SimFile.reset_index()
    GroupsFileMod=GroupsFile.reset_index().rename(columns={'Sensor' : 'index'})
    GroupsOnce=GroupsFileMod.drop_duplicates('index')
    s1=SimFileInd['index']
    s2=GroupsOnce['index']
    TagIntersection=s1[s1.isin(s2)]
    SubSimFile=SimFile.loc[TagIntersection.to_numpy(),TagIntersection.to_numpy()]
    SubSimNp=SubSimFile.to_numpy()
    SubSimFile=SubSimFile.reset_index()
    BigMerge=pd.merge(SubSimFile,GroupsOnce,on='index',how='inner')
    TagGroups=BigMerge[['index','Group']]
    TagGroupSeries=np.array(TagGroups['Group'])
    TagList=np.array(TagGroups['index'])
    TagGroupsUnique=TagGroups.drop_duplicates('Group')
    TagGroupsUniqueSeries=np.array(TagGroupsUnique['Group'])
    np.fill_diagonal(SubSimNp,np.nan)
    SimGroups=pd.DataFrame(SubSimNp,index=TagGroupSeries,columns=TagGroupSeries)
    SimFileOut=SubSimFile
    TagGroupsUniqueSeriesPD=pd.DataFrame(TagGroupsUniqueSeries)
    TagListPD=pd.DataFrame(TagList)
    
    return SimGroups, TagGroupsUniqueSeriesPD, TagGroups, TagListPD, SimFileOut

def CreateGroupSizeCount(TagGroups):
    Singles=pd.Series([TagGroups.Group.str.count('Single').sum()],index=['Single'])
    GroupSizeCount=TagGroups.Group.value_counts()
    GroupSizeCount.loc['Single']=np.nan
    GroupHist=GroupSizeCount.value_counts()
    GroupHistSingles=GroupHist.append(Singles)
    return GroupHistSingles

def CalculateGroupSimMatrixMean(SimGroups,TagGroupsUniqueSeries):
    df=SimGroups
    rc=list(TagGroupsUniqueSeries[0])
    sums=np.zeros((len(rc),len(rc)))
    result=pd.DataFrame(sums,index=rc,columns=rc)
    for r in rc:
        for c in rc:
            subdf=df.loc[r,c]
            if subdf.size > 1:
                result.loc[r,c]=np.nanmean(subdf.values)
            else:
                result.loc[r,c]=subdf

    result=result.round(2)
    return result

def TagsInGroup(GroupName,TagGroups):
    Matches=TagGroups[TagGroups.Group == GroupName]
    return Matches['index']

def GroupForTag(TagName,TagGroups):
    Matches=TagGroups[TagGroups['index'] == TagName]
    return Matches['Group']

def CreateSubMatrixForGroup(GroupName,GroupSimMatrix,TopX):        
    if TopX > len(GroupSimMatrix): 
        TopX = len(GroupSimMatrix)
    SortedMatrix=GroupSimMatrix.sort_values(GroupName, ascending=False)
    SortedIndex=SortedMatrix.index
    TopGroups=SortedIndex[:TopX]
    SubMatrix=SortedMatrix.loc[TopGroups,TopGroups]
    SubMatrixOut = SubMatrix.round(2)
    
    return SubMatrixOut

def CreateSubMatrixForTag(TagName,SimFile,TopX=10):        
    if TopX > len(SimFile): 
        TopX = len(SimFile)
    SimFile = SimFile.set_index('index')
    SortedMatrix=SimFile.sort_values(TagName, ascending=False)
    SortedIndex=SortedMatrix.index
    TopTags=SortedIndex[:TopX]
    SubMatrix=SortedMatrix.loc[TopTags,TopTags]
    SubMatrixOut = SubMatrix.round(2)
    return SubMatrixOut

# def CalculateGroupSimMatrixMean(SimGroups,TagGroupsUniqueSeries):
#     df=SimGroups
#     rc=list(TagGroupsUniqueSeries[0])
#     sums=np.zeros((len(rc),len(rc)))
#     result=pd.DataFrame(sums,index=rc,columns=rc)
#     for r in rc:
#         for c in rc:
#             subdf=df.loc[r,c]
#             if subdf.size > 1:
#                 result.loc[r,c]=np.nanmean(subdf.values)
#             else:
#                 result.loc[r,c]=subdf

#     result=result.round(2)
#     return result

def CalculateGroupSimKPIs(SimGroups,TagGroupsUniqueSeries,GroupSimMatrixMean):
    df=SimGroups
    dfgm=GroupSimMatrixMean
    rc=list(TagGroupsUniqueSeries[0])
    sums=np.zeros((len(rc),6))
    result=pd.DataFrame(sums,index=rc,columns=['Mean','Min','Max','MaxOut','MaxGmean','Size'])
    for r in rc:
        c=r
        #for c in rc:
        subdf=df.loc[r,c]
        if subdf.size > 1:
            result.loc[r,'Mean']=np.nanmean(subdf.values)
            result.loc[r,'Min']=np.nanmin(subdf.values)
            result.loc[r,'Max']=np.nanmax(subdf.values)
            result.loc[r,'Size']=len(subdf)
        else:
            result.loc[r,'Mean']=1
            result.loc[r,'Min']=1
            result.loc[r,'Min']=1
            result.loc[r,'Size']=1
        
        #subdfn=df.loc[r,set(list(TagGroupsUniqueSeries))-set([c])]
        #subdfgmn=dfgm.loc[r,set(list(TagGroupsUniqueSeries))-set([c])]
        subdfn=df.loc[r,set(rc)-set([c])]
        subdfgmn=dfgm.loc[r,set(rc)-set([c])]
        result.loc[r,'MaxOut']=np.nanmax(subdfn.values)
        result.loc[r,'MaxGmean']=np.nanmax(subdfgmn.values)
    result.reset_index(inplace = True)
    result = result.rename(columns = {'index': 'Group'})
    result=result.round(2)
    return result

def CreateTopTagsForTag(TagName,SimFileOut,TagGroups,TopX):        
    if TopX > len(SimFileOut): 
        TopX = len(SimFileOut)
    Combined=pd.merge(SimFileOut,TagGroups,on='index')
    SortedMatrix=Combined.sort_values(TagName, ascending=False)
    HelpMatrix=SortedMatrix[:TopX]
    SubMatrix=HelpMatrix[['index',TagName,'Group']]
    
    return SubMatrix

def SimMatrixForGroup(GroupName, TagGroups, SimFileOut, TopX=15):
    if TopX > len(SimFileOut): 
        TopX = len(SimFileOut)
    TagsInGroup = TagGroups[TagGroups.Group == GroupName]
    TagsInGroup = list(TagsInGroup['index'])
    
    GroupMatrix = SimFileOut[SimFileOut['index'].isin(TagsInGroup)] #[TagsInGroup]
    GroupMatrix = GroupMatrix[TagsInGroup]
    GroupMatrix.reset_index(drop=True, inplace = True)
    GroupMatrix['Tag'] = TagsInGroup
    GroupMatrix = GroupMatrix.set_index('Tag')
    GroupMatrix = GroupMatrix.round(2)
    GroupMatrix.sort_values(GroupMatrix.columns[0], axis=1, ascending = False, inplace = True)
    GroupMatrix.sort_values(GroupMatrix.columns[0], ascending = False, inplace = True)
    
    return GroupMatrix

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'Taglist' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=',', index_col= 0)
        elif 'SimilarityMatrix' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';', index_col= 0, decimal=",")
    except Exception as e:
        print(e)
        return None
    return df

@app.callback([Output('tablediv', 'children'),
              Output('tablediv2', 'children')],
             [Input('kpitable', 'derived_virtual_data')])

def make_table_graphs(rows):
    if rows is not None:
        df = pd.DataFrame.from_dict(rows)
        cols = list(pd.DataFrame(rows))
        xdata = list(df[cols[0]])
        means = list(df[cols[1]])
        mins = list(df[cols[2]])
        maxs = list(df[cols[3]])
        maxout = list(df[cols[4]])
        maxgmean = list(df[cols[5]])
        sizes = list(df[cols[6]])
        scatterplot1 = html.Div(dcc.Graph(id='scatterplot1',figure={
                    "data": [go.Bar(x = xdata, y = sizes, yaxis = 'y2', name = "Group size", opacity = 0.5),
                             go.Scatter(x = xdata, y = maxs, mode='markers', marker = dict(size=10, color = "red"), name = "Group Max"),
                             go.Scatter(x = xdata, y = mins, mode='markers', marker = dict(size=10, color = "red"), name = "Group Min"),
                             go.Scatter(x = xdata, y = means, mode='markers', marker = dict(size=10, color = "black"), name = "Group Mean")],
                    "layout": {
                        'title': "Group Statistics",
                        'yaxis2': {'overlaying':'y', 'side':'right'},
                        'margin': {'t':'50', 'l':'75', 'r':'10', 'b':'75'},
                        'width': 1400,
                        'height': 400
                    }}),className = 'row')
        
        scatterplot2 = html.Div(dcc.Graph(id='scatterplot2',figure={
                    "data": [go.Scatter(x = xdata, y = maxgmean, mode='markers', marker = dict(size=10, color = "green"), name = "Max Gr Sim"),
                             go.Scatter(x = xdata, y = maxout, mode='markers', marker = dict(size=10, color = "blue"), name = "Max TagSim"),],
                    "layout": {
                        'title': "Max Similarities to outer Tag and Group",
                        'margin': {'t':'50', 'l':'75', 'r':'10', 'b':'75'},
                        'width': 1400,
                        'height': 400
                    }}),className = 'row')
        return [scatterplot1], [scatterplot2]
    return [None, None]
    

@app.callback(Output('similaritymatrixoutput', 'children'),
              [Input('similaritymatrix', 'contents'),
               Input('similaritymatrix', 'filename')])
def read_uploaded_similaritymatrix(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        json_similaritymatrix = {
            'SimFile': df.to_json(orient='split', date_format='iso')
        }
        if df is not None:
            return json.dumps(json_similaritymatrix)
        else:
            return None
    else:
        return None

@app.callback(Output('taglistoutput', 'children'),
              [Input('taglist', 'contents'),
               Input('taglist', 'filename')])
def read_uploaded_taglist(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        json_taglist = {
            'GroupsFile': df.to_json(orient='split', date_format='iso')
        }
        if df is not None:
            return json.dumps(json_taglist)
        else:
            return None
    else:
        return None
    
@app.callback(Output('output', 'children'),
              [Input('taglistoutput', 'children'),
              Input('similaritymatrixoutput', 'children')])
def update_output(json_taglist, json_similaritymatrix):
    if json_taglist is not None:
        if json_similaritymatrix is not None:
            json_taglist = json.loads(json_taglist)
            GroupsFile = pd.read_json(json_taglist['GroupsFile'], orient='split')
            json_similaritymatrix = json.loads(json_similaritymatrix)
            SimFile = pd.read_json(json_similaritymatrix['SimFile'], orient='split')
            SimGroups, TagGroupsUniqueSeries, TagGroups, TagList, SimFileOut = ModifyInputFiles(SimFile,GroupsFile)
            TagGroupsUniqueSeries = pd.DataFrame(TagGroupsUniqueSeries)
            TagList = pd.DataFrame(TagList)
            GroupSimMatrixMean = CalculateGroupSimMatrixMean(SimGroups,TagGroupsUniqueSeries)
            GroupSimMatrixMean = pd.DataFrame(GroupSimMatrixMean)
            json_datasets = {
                'SimGroups': SimGroups.to_json(orient='split', date_format='iso'),
                'TagGroupsUniqueSeries': TagGroupsUniqueSeries.to_json(orient='split', date_format='iso'),
                'TagGroups': TagGroups.to_json(orient='split', date_format='iso'),
                'TagList': TagList.to_json(orient='split', date_format='iso'),
                'SimFileOut': SimFileOut.to_json(orient='split', date_format='iso'),
                'GroupSimMatrixMean': GroupSimMatrixMean.to_json(orient='split', date_format='iso')
            }
            SimFileOut_cols = list(SimFileOut.columns)
            return json.dumps(json_datasets)

@app.callback(Output('groupkpis', 'children'),
              [Input('output', 'children')])

def create_kpi_tables(json_dataset):
    if json_dataset is not None:
        datasets = json.loads(json_dataset)
        SimGroups = pd.read_json(datasets['SimGroups'], orient='split')
        TagGroupsUniqueSeries = pd.read_json(datasets['TagGroupsUniqueSeries'], orient='split')
        GroupSimMatrixMean = pd.read_json(datasets['GroupSimMatrixMean'], orient='split')
        groupsimkpis = CalculateGroupSimKPIs(SimGroups,TagGroupsUniqueSeries,GroupSimMatrixMean)
        data = groupsimkpis.to_dict('rows')
        columns = [{"name": i, "id": i,} for i in (groupsimkpis.columns)]
        table = html.Div([dt.DataTable(id = 'kpitable',data=data, columns=columns, style_cell={'textAlign': 'left'},
                                       sort_action='native', sort_mode='multi')], style={'height': 800, 'overflowY': 'scroll', 'overflowX': 'scroll'})
        return table
    else:
        return [None]
        
@app.callback(Output('group dropdown', 'options'),
              [Input('output', 'children')])

def update_dropdown(json_dataset):
    if json_dataset is not None:
        datasets = json.loads(json_dataset)
        TagGroupsUniqueSeries = pd.read_json(datasets['TagGroupsUniqueSeries'], orient='split')
        columns = TagGroupsUniqueSeries[0].unique()
        if columns is not None:
            return [{'label': x, 'value': x} for x in columns]
        else:
            return []
    else:
        return []

@app.callback([Output('tags for group', 'children'),
              Output('tags for group title', 'children')],
              [Input('group dropdown', 'value'),
              Input('output', 'children')])

def update_tags_for_group(group, json_dataset):
    if json_dataset is not None:
        if group is not None:
            GroupName = group
            datasets = json.loads(json_dataset)
            TagGroups = pd.read_json(datasets['TagGroups'], orient='split')
            SimFileOut = pd.read_json(datasets['SimFileOut'], orient='split')
            TopX = 15
            GroupMatrix = SimMatrixForGroup(GroupName, TagGroups, SimFileOut, TopX=TopX)
            
            columns = list(GroupMatrix.columns)
            values = GroupMatrix.values
            annotations = []

            for n, row in enumerate(values):
                for m, val in enumerate(row):
                    annotations.append(go.layout.Annotation(text=str(values[n][m]), x=columns[m], y=columns[n],
                                                            xref='x1', yref='y1', showarrow=False, font=dict(color="#ffffff", size=12)))
            
            groupheatmap = html.Div(dcc.Graph(id='group heatmap',figure={
                "data": [go.Heatmap(z=values, x=columns, y=columns, colorscale = colorscale)],
                "layout": {
                    'annotations': annotations,
                    'yaxis': {'autorange': 'reversed'},
                    'xaxis': {'side': 'top'},
                    'margin': {'t':'100', 'l':'150', 'r':'10', 'b':'10'},
                    'width': 900,
                    'height': 900
                }}),className = 'row')
            
            title = html.Div(html.H2("Group Similarity Matrix"))
            
            return groupheatmap, title
        else:
            return [None, None]
    else:
        return [None, None]

    
@app.callback(Output('tag dropdown', 'options'),
              [Input('output', 'children')])

def update_dropdown2(json_dataset):
    if json_dataset is not None:
        datasets = json.loads(json_dataset)
        TagList = pd.read_json(datasets['TagList'], orient='split')
        columns = TagList[0].unique()
        if columns is not None:
            return [{'label': x, 'value': x} for x in columns]
        else:
            return []
    else:
        return []

@app.callback([Output('tagsimmatrix', 'children'),
              Output('tagheatmap','children'),
              Output('heatmaptitle', 'children')],
              [Input('tag dropdown', 'value'),
              Input('output', 'children')])

def update_group_for_tag(tag, json_dataset):
    if json_dataset is not None:
        if tag is not None:
            TagName = tag
            datasets = json.loads(json_dataset)
            TagGroups = pd.read_json(datasets['TagGroups'], orient='split')
            SimFileOut = pd.read_json(datasets['SimFileOut'], orient='split')
            TopX = 15
            Tags = len(TagGroups)
            
            GroupName = pd.DataFrame(GroupForTag(TagName, TagGroups))
            GroupName = GroupName['Group'].iloc[0]
            GroupTagList = list(TagsInGroup(GroupName, TagGroups))

            SubMatrix = CreateTopTagsForTag(TagName,SimFileOut,TagGroups,Tags)
            data1 = SubMatrix.to_dict('rows')
            columns1 = [{"name": i, "id": i,} for i in (SubMatrix.columns)]
            table1 = html.Div([dt.DataTable(data=data1, columns=columns1, style_cell={'textAlign': 'left'}, sort_action='native', sort_mode='multi')],
                              style={'height': 800, 'overflowY': 'scroll', 'overflowX': 'scroll', 'marginTop': 150})
            SubTagMatrix = CreateSubMatrixForTag(TagName, SimFileOut, TopX)
            columns = list(SubTagMatrix.columns)
            values = SubTagMatrix.values
            annotations = []

            for n, row in enumerate(values):
                for m, val in enumerate(row):
                    if columns[m] in GroupTagList and columns[n] in GroupTagList and GroupName != 'Single':
                        annotations.append(go.layout.Annotation(text=str(values[n][m]), x=columns[m], y=columns[n],
                                                                xref='x1', yref='y1', showarrow=False, font=dict(color="green", size=14)))
                    else:
                        annotations.append(go.layout.Annotation(text=str(values[n][m]), x=columns[m], y=columns[n],
                                                                xref='x1', yref='y1', showarrow=False, font=dict(color="#ffffff", size=12)))

            tagheatmap = html.Div(dcc.Graph(id='tag heatmap',figure={
                "data": [go.Heatmap(z=values, x=columns, y=columns, colorscale = colorscale)],
                "layout": {
                    'annotations': annotations,
                    'yaxis': {'autorange': 'reversed'},
                    'xaxis': {'side': 'top'},
                    'margin': {'t':'100', 'l':'200', 'r':'30', 'b':'10'},
                    'width': 1200,
                    'height': 1000
                }}),className = 'row')
            
            heatmaptitle = html.Div(html.H2("Top Similarities for the Tag"))
            
            return table1, tagheatmap, heatmaptitle
        else:
            return [None, None, None]
    else:
        return [None, None, None]
    
@app.callback(
    Output(component_id='testdiv', component_property='children'),
    [Input('group dropdown', 'value'),
    Input('output', 'children')])

def make_subgroupsimmatrix(group, json_dataset):
    if json_dataset is not None:
        if group is not None:
            datasets = json.loads(json_dataset)
            size = 15
            SimGroups = pd.read_json(datasets['SimGroups'], orient='split')
            TagGroupsUniqueSeries = pd.read_json(datasets['TagGroupsUniqueSeries'], orient='split')
            GroupSimMatrixMean = pd.read_json(datasets['GroupSimMatrixMean'], orient='split')
            SubGroupSimMatrix=CreateSubMatrixForGroup(group,GroupSimMatrixMean,size)
            json_dataset_subgroup = {
                'SubGroupSimMatrix': SubGroupSimMatrix.to_json(orient='split', date_format='iso')
            }
            return json.dumps(json_dataset_subgroup)

    
@app.callback(
    [Output(component_id='graafin title', component_property='children'),
    Output(component_id='graafi', component_property='children')],
    [Input('testdiv', 'children')])    

def update_graph(json_dataset):
    if json_dataset is not None:
        dataset = json.loads(json_dataset)
        SubGroupSimMatrix = pd.read_json(dataset['SubGroupSimMatrix'], orient='split')
        columns = list(SubGroupSimMatrix.columns)
        values1 = SubGroupSimMatrix.values
        values = list(np.around(np.array(values1),2))
        annotations = []
        for n, row in enumerate(values):
            for m, val in enumerate(row):
                annotations.append(go.layout.Annotation(text=str(values[n][m]), x=columns[m], y=columns[n],
                                                        xref='x1', yref='y1', showarrow=False, font=dict(color="#ffffff", size=12)))
        graafi = html.Div(dcc.Graph(
                    id='groupheatmap2',
                    figure={
                        "data": [go.Heatmap(z=values,
                                            x=columns,
                                            y=columns,
                                            colorscale = colorscale)
                        ],
                        "layout": {
                            'annotations': annotations,
                            'yaxis': {'autorange': 'reversed', 'automargin': True},
                            'xaxis': {'side': 'top', 'automargin': True},
                            'autosize': True,
                            'margin': {'t':'50', 'l':'100', 'r':'0', 'b':'0'},
                            'width': 900,
                            'height': 900
                        },
                    },
                ),
            className = 'row'
        )
        title = html.Div(html.H2("Top Groups for the selected Group"))
        
        return title, graafi
        
    return [None, None]
        
if __name__ == '__main__':
    app.run_server(port=5001)