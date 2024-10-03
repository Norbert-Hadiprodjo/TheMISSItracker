import plotly.graph_objects as go

# Sample data
students = ['Student A', 'Student B', 'Student C']
stroke_speeds_seconds = [0.8, 0.7, 0.9]  # Seconds per stroke
stroke_rates = [30, 32, 28]  # Strokes per minute
distances_covered = [1000, 1100, 950]  # Meters covered

# Create figure with subplots
fig = go.Figure()

# Add bar trace for stroke speed
fig.add_trace(go.Bar(x=students, y=stroke_speeds_seconds, name='Stroke Speed', marker_color='skyblue'))

# Add scatter trace for stroke rate
fig.add_trace(go.Scatter(x=students, y=stroke_rates, mode='markers', name='Stroke Rate (SPM)', marker=dict(color='orange', size=12)))

# Add scatter trace for distance covered
fig.add_trace(go.Scatter(x=students, y=distances_covered, mode='lines+markers', name='Distance Covered (m)', line=dict(color='green', width=2)))

# Customize layout
fig.update_layout(
    title='Canoeing Statistics of Students',
    xaxis_title='Students',
    yaxis_title='Metric',
    yaxis=dict(range=[0, max(max(stroke_speeds_seconds), max(stroke_rates), max(distances_covered)) + 100]),
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    font=dict(color='black'),
    margin=dict(l=50, r=50, t=50, b=50),
    legend=dict(orientation='h', x=0.5, y=-0.15)
)

# Show the plot
fig.show()
