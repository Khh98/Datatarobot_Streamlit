import copy
from typing import Optional, Any

import pandas as pd
import streamlit as st
import altair as alt
from altair import Undefined
from datarobot import TARGET_TYPE
import plotly.express as px

from .caches import get_project, get_model, initialize_and_get_feature_impact
from .select_box import AGGREGATED_NAME
from .wrappers import chart_with_error_backup




@st.cache
def _get_aggregated_feature_impact_data(project_id: str, model_id: str):
    fi_data = initialize_and_get_feature_impact(project_id=project_id, model_id=model_id, use_multiclass=False)
    return pd.DataFrame.from_records(fi_data).sort_values('impactNormalized', ignore_index=True)


@st.cache
def _get_multiclass_feature_impact_data(project_id: str, model_id: str):
    aggregated_fi = _get_aggregated_feature_impact_data(project_id=project_id, model_id=model_id,)
    feature_impacts = [{'class': AGGREGATED_NAME, 'featureImpacts': aggregated_fi}]
    mc_fi_data = copy.copy(initialize_and_get_feature_impact(project_id=project_id, model_id=model_id, use_multiclass=True))
    for entry in mc_fi_data:
        entry['featureImpacts'] = pd.DataFrame.from_records(entry['featureImpacts'])
        feature_impacts.append(entry)
    return feature_impacts


@chart_with_error_backup
def derived_features_chart(project_id: str, model_id: str, selected_class: Optional[str] = None, height: Any = None, width: Any = None):
    project = get_project(project_id)
    model = get_model(project_id=project_id, model_id=model_id)
    if project.target_type == TARGET_TYPE.MULTICLASS:
        feature_impacts = _get_multiclass_feature_impact_data(project_id=project_id, model_id=model_id)
        fig = px.bar(next(c['featureImpacts'] for c in feature_impacts if c['class'] == selected_class),
                     y='featureName',
                     x='impactNormalized',
                     title=model.model_type,
                     height=400,
                     width=500,
                     )
        return st.plotly_chart(fig)
    else:
        feature_impact = _get_aggregated_feature_impact_data(project_id=project_id, model_id=model_id)
        # Convert feature_impact to a DataFrame 
        feature_impact_df = pd.DataFrame(feature_impact)
    # Sort by 'impactNormalized' and get the top 5 features
        feature_impact_df = feature_impact_df.sort_values(by='impactNormalized', ascending=False)
        feature_impact_df['color'] = ['rgba(222,45,38,0.8)' if i < 5 else 'rgba(204,204,204,0.8)' for i in range(len(feature_impact_df))]
        fig = px.bar(feature_impact_df,
                 y='featureName',
                 x='impactNormalized',
                 color='color',
                 color_discrete_map="identity",
                 title=model.model_type,
                 height=600,
                 width=500,
                 )
        return st.plotly_chart(fig)