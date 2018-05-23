from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

with open('../data/df.pkl','rb') as f:
    df = pickle.load(f)

with open('../data/df_firm_zip_lookup.csv') as f:
    df_firm_zip = pd.read_csv(f)

with open('../data/free-zipcode-database-Primary.csv') as f:
    df_zip = pd.read_csv(f)


features = ['top_elev_(ft)',
            'bottom_elev_(ft)',
            'vert_rise_(ft)',
            'slope_length_(ft)',
            'avg_width_(ft)',
            'slope_area_(acres)',
            'avg_grade_(%)',
            'max_grade_(%)',
            'groomed']
X = df[features].values
ss = StandardScaler()
X = ss.fit_transform(X)


def cos_sim_recs(index, n=5, resort=None, color=None):
    trail = X[index].reshape(1,-1)
    cs = cosine_similarity(trail, X)
    rec_index = np.argsort(cs)[0][::-1][1:]
    ordered_df = df.loc[rec_index]
    if resort:
        ordered_df = ordered_df[ordered_df['resort'] == resort]
    if color:
        ordered_df = ordered_df[ordered_df['colors'].isin(color)]
    rec_df = ordered_df.head(n)
    rec_df = rec_df.reset_index(drop=True)
    rec_df.index = rec_df.index+1
    orig_row = df.loc[[index]].rename(lambda x: 'original')
    total = pd.concat((orig_row,rec_df))
    return total

def clean_df_for_recs(df):
    df['groomed'][df['groomed'] == 1] = 'Groomed'
    df['groomed'][df['groomed'] == 0] = 'Ungroomed'
    df['color_names'] = df['colors']
    df['color_names'][df['color_names'] == 'green'] = 'Green'
    df['color_names'][df['color_names'] == 'blue'] = 'Blue'
    df['color_names'][df['color_names'] == 'black'] = 'Black'
    df['color_names'][df['color_names'] == 'bb'] = 'Double Black'
    df = df[['trail_name','resort','location','color_names','groomed','top_elev_(ft)','bottom_elev_(ft)','vert_rise_(ft)','slope_length_(ft)','avg_width_(ft)','slope_area_(acres)','avg_grade_(%)','max_grade_(%)']]
    df.columns = ['Trail Name', 'Resort','Location','Difficulty','Groomed','Top Elev (ft)', 'Bottom Elev (ft)', 'Vert Rise (ft)', 'Slope Length (ft)', 'Avg Width (ft)', 'Slope Area (acres)', 'Avg Grade (%)', 'Max Grade (%)']
    return df


@app.route('/', methods =['GET','POST'])
def index():
    return render_template('home.html')

@app.route('/fund_recommender', methods=['GET','POST'])
def trails():
    return render_template('index.html',df=df, df_firm_zip = df_firm_zip, df_zip = df_zip)

@app.route('/fund_results/<int:office_id>', methods=['GET'])
def fund_results(office_id):

    # Recommendation 1: top_n_1.p
    # Recommendation 2: top_n_2.p
    # Office 1: office_desc_1.p
    # Office 2: office_desc_2.p

    with open('../data/office_desc_{}.p'.format(office_id), 'rb') as f:
        office_description = pickle.load(f)

    with open('../data/top_n_{}.p'.format(office_id), 'rb') as f:
        fund_recommendations = pickle.load(f)

    return render_template('fund_results.html', office_description=office_description, fund_recommendations=fund_recommendations)

@app.route('/recommendations', methods=['GET','POST'])
def recommendations():
    color_lst = None
    if request.form.get('green'):
        color_lst = ['green']
    if request.form.get('blue'):
        color_lst = ['green','blue']
    if request.form.get('black'):
        color_lst = ['green','blue','black']
    if request.form.get('bb'):
        color_lst = ['green','blue','black','bb']
    # CHECKBOX FUNCTIONALITY!!!
    resort = request.form['resort']
    if resort == '':
        return 'You must select a trail from your favorite resort.'
    trail = request.form['trail']
    if trail != '':
        index = int(trail)
        dest_resort = request.form['dest_resort']
        num_recs = int(request.form['num_recs'])
        rec_df = cos_sim_recs(index,num_recs,dest_resort,color_lst)
        rec_df = clean_df_for_recs(rec_df)
        if dest_resort == '':
            resort_links = links[resort]
        else:
            resort_links = links[dest_resort]
        return render_template('recommendations.html',rec_df=rec_df,resort_links=resort_links)
    return 'You must select a trail.'

@app.route('/get_trails')
def get_trails():
    resort = request.args.get('resort')
    if resort:
        sub_df = df[df['resort'] == resort]
        sub_df['trail_name'] = sub_df['trail_name'].apply(lambda x: x.split()).apply(lambda x: (x[1:] + ['Upper']) if (x[0] == 'Upper') else x).apply(lambda x: ' '.join(x))
        sub_df['trail_name'] = sub_df['trail_name'].apply(lambda x: x.split()).apply(lambda x: (x[1:] + ['Lower']) if (x[0] == 'Lower') else x).apply(lambda x: ' '.join(x))
        sub_df.sort_values(by='trail_name',inplace=True)
        id_name_color = [("","Select a Trail...","white")] + list(zip(list(sub_df.index),list(sub_df['trail_name']),list(sub_df['colors'])))
        data = [{"id": str(x[0]), "name": x[1], "color": x[2]} for x in id_name_color]
        # print(data)
    return jsonify(data)

@app.route('/trail_map/<resort>')
def trail_map(resort):
    resort_image = links[resort][0]
    return render_template('trail_map.html',resort_image=resort_image)

if  __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
