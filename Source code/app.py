from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import pickle,re,numpy as np
import requests
import time

app = Flask(__name__)
app.secret_key = 'Dhaha'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    mobile_number = db.Column(db.String(10), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

def delete_all_users():
    try:
        # This deletes all records in the User table
        num_rows_deleted = db.session.query(User).delete()
        db.session.commit()
        return f"Deleted {num_rows_deleted} users."
    except Exception as e:
        db.session.rollback()
        return f"An error occurred: {e}"
    
with open('top_rated.pkl', 'rb') as f:
    top_rated = pickle.load(f)

with open('latest_movies.pkl', 'rb') as f:
    latest_movies = pickle.load(f)

with open('data.pkl', 'rb') as f:
    movie_list = pickle.load(f)

with open('movies_data.pkl', 'rb') as f:
    movies_data = pickle.load(f)

with open('similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)


with open('user_movie_matrix.pkl', 'rb') as f:
    user_movie_matrix = pickle.load(f)



from requests.exceptions import ConnectionError, HTTPError
import time

MAX_RETRIES = 3

def fetch_movie_details(movie_title):
    clean_title = re.sub(r"(\s*\(.*?\)\s*|\s*\[.*?\]\s*)", "", movie_title).strip()
    api_key = 'c06cede91056be933f6b25e34556347f'
    url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data['results']:
            movie_data = data['results'][0]
            movie_details = {
                'title': movie_data.get('title', movie_title),
                'release_year': movie_data.get('release_date', '').split('-')[0],
                'overview': movie_data.get('overview', ''),
                'tagline': movie_data.get('tagline', ''),
                'original_language': movie_data.get('original_language', ''),
                'runtime': f"{movie_data.get('runtime', '')} minutes",
                'rating': movie_data.get('vote_average', ''),
                'poster_path': f'https://image.tmdb.org/t/p/original{movie_data.get('poster_path')}',
                'trailer_url': get_movie_trailer_url(movie_data.get('id'))
            }
        else:
            movie_details = {
                'title': movie_title,
                'release_year': '',
                'overview': '',
                'tagline': '',
                'original_language': '',
                'runtime': '',
                'rating': '',
                'poster_path': '',
                'trailer_url': ''
            }
        
        return movie_details
    except (ConnectionError, HTTPError) as e:
        print(f"Error fetching movie details: {e}")
        return {
            'title': movie_title,
            'release_year': '',
            'overview': '',
            'tagline': '',
            'original_language': '',
            'runtime': '',
            'rating': '',
            'poster_path': '',
            'trailer_url': ''
        }

def fetch_movie_poster(movie_title):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            clean_title = re.sub(r"(\s*\(.*?\)\s*|\s*\[.*?\]\s*)", "", movie_title).strip()
            api_key = 'c06cede91056be933f6b25e34556347f'
            url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data['results']:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return f'https://image.tmdb.org/t/p/original{poster_path}'
            else:
                return f'https://www.google.com/{movie_title}.jpg'
        except (ConnectionError, HTTPError) as e:
            print(f"Error fetching movie poster: {e}")
            retries += 1
            if retries < MAX_RETRIES:
                print(f"Retrying... ({retries}/{MAX_RETRIES})")
                time.sleep(1)  # Wait for a second before retrying
            else:
                print(f"Max retries reached. Unable to fetch movie poster for {movie_title}.")
                return None

    return None

def get_movie_trailer_url(movie_id):
    api_key = 'c06cede91056be933f6b25e34556347f'
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}'
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if data['results']:
        for video in data['results']:
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                return f'https://www.youtube.com/watch?v={video["key"]}'
    return ''



# Initialize KNN model
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

knn_model = NearestNeighbors(n_neighbors=20, metric='cosine')  # Set n_neighbors to 100
knn_model.fit(similarity_matrix)


def recommend_movies_based_on_movie(movie_id, num_recommendations=15):
    # Check if movie_id is present in the user_movie_matrix DataFrame
    if movie_id not in user_movie_matrix.columns:
        print(f"Movie ID {movie_id} not found in the dataset.")
        return []

    users_who_watched = user_movie_matrix[user_movie_matrix[movie_id] > 0].index.tolist()
    
    # Get the subset of the matrix corresponding to these users
    subset_matrix = user_movie_matrix.loc[users_who_watched]
    
    # Count how many times each movie was watched by these users
    movie_popularity = subset_matrix.apply(np.sum, axis=0)
    
    # Remove the original movie from the list
    movie_popularity = movie_popularity.drop(movie_id)
    
    # Sort movies by the most watched
    recommended_movies = movie_popularity.sort_values(ascending=False).head(num_recommendations).index
    
    # Get movie titles
    recommended_movie_titles = movies_data[movies_data['movieId'].isin(recommended_movies)]['title'].tolist()
    
    return recommended_movie_titles


def get_recommendations(movie_id):
    # Find index of movie in the dataset
    idx = movies_data[movies_data['movieId'] == movie_id].index[0]
    
    # Get K nearest neighbors
    _, neighbor_indices = knn_model.kneighbors([similarity_matrix[idx]])
    
    # Get movie IDs of nearest neighbors
    neighbor_movie_ids = movies_data.iloc[neighbor_indices[0]]['movieId']
    
    # Get movie names of nearest neighbors
    neighbor_movie_names = movies_data.iloc[neighbor_indices[0]]['title']
    
    # Create DataFrame with movie IDs and names
    recommendations_df = pd.DataFrame({'MovieID': neighbor_movie_ids.values, 'MovieName': neighbor_movie_names.values})
    
    # Drop duplicate movie IDs
    recommendations_df = recommendations_df.drop_duplicates(subset=['MovieID'])

    movie=recommendations_df['MovieName'].values
    movie=movie.tolist()
    
    return movie



with app.app_context():
    db.create_all()

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('signup.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile_number = request.form['mobile_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if the email already exists in the database
        user_exists = User.query.filter_by(email=email).first()
        num_exist = User.query.filter_by(mobile_number=mobile_number).first()
        if user_exists:
            return render_template('signup.html', error="Email already exists.")
        if num_exist: 
            return render_template('signup.html', error="Mobile number already exists.")

        if password != confirm_password:
            return render_template('signup.html', error="Password and confirm password do not match.")

        # No existing user and passwords match, proceed to create new user
        user = User(name=name, email=email, mobile_number=mobile_number, password=password)
        try:
            db.session.add(user)
            db.session.commit()
        except sqlalchemy.exc.IntegrityError as e:
            db.session.rollback()
            return render_template('signup.html', error="An error occurred, please try again.")

        session['name'] = user.name
        return redirect(url_for('home'))

    return render_template('signup.html')


@app.route('/signin', methods=['GET','POST'])
def signin():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if email is None or password is None:
            error = "Invalid email or password"
            return render_template('signin.html', error=error)

        # Check if the email exists in the database
        user = User.query.filter_by(email=email).first()
        passwrd = User.query.filter_by(email=email,password=password).first()
        if user:
            # Email exists, now check if the password matches
            if passwrd:
                session['name'] = user.name
                # Password matches, navigate to home page
                return redirect(url_for('home'))
            else:
                # Password does not match
                error = "Incorrect Password"
                return render_template('signin.html', error=error)
        else:
            # Email not found in the database
            error = "Email not found, Signup required"
            return render_template('signin.html', error=error)

    # If the request method is GET, render the signin.html template
    return render_template('signin.html')



# Route for home page
@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'name' in session:
        username = session['name']
    else:
        # Handle the case where 'name' is not in the session
        return redirect(url_for('signin'))
    # Fetch posters for top-rated movies
    top_rated_posters = {}
    for title in top_rated:  # Assume top_rated is a list of titles
        poster_url = fetch_movie_poster(title)
        top_rated_posters[title] = poster_url

    # Fetch posters for latest movies
    latest_movies_posters = {}
    for title in latest_movies:  # Assume latest_movies is a list of titles
        poster_url = fetch_movie_poster(title)
        latest_movies_posters[title] = poster_url

    return render_template('home.html', username=username, top_rated=top_rated, latest_movies=latest_movies,
                           top_rated_posters=top_rated_posters, latest_movies_posters=latest_movies_posters,movie_list=movie_list)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    print(request.method)
    if request.method == 'POST' or request.method == 'GET':
        print(":")
        movie_title = request.args.get('sb')
        print(movie_title)
        
        # Ensure movie_title is a string
        if not isinstance(movie_title, str):
            movie_title = str(movie_title)
        
        # Fetch movieId based on movie title
        movie_det = movies_data[movies_data['title'] == movie_title]
        if len(movie_det) > 0:
            movie_id = movie_det['movieId'].values[0]
        else:
            print("Movie not found in the dataset.")
            return "Movie not found in the dataset.", 404
        movie_details=fetch_movie_details(movie_title)
        print(movie_details)
        poster_url = fetch_movie_poster(movie_title)
        # Get recommendations based on movie ID (replace this with your logic)
        recommendations = get_recommendations(movie_id)
        
        # Get recommendations for the user (replace this with your logic)
        recommended_for_user = recommend_movies_based_on_movie(movie_id)
        
        # Prepare movie posters for recommended movies
        recommended_movies = {}
        for title in recommendations:
            poster_url = fetch_movie_poster(title)
            recommended_movies[title] = poster_url
        
        # Prepare movie posters for movies recommended for the user
        recommended_for_user_posters = {}
        for title in recommended_for_user:
            poster_url = fetch_movie_poster(title)
            recommended_for_user_posters[title] = poster_url

        
        return render_template('recommend.html', movie_details=movie_details, recommended_movies=recommended_movies, recommended_for_user=recommended_for_user_posters, movie_title=movie_title,movie_list=movie_list,poster_url=poster_url)



if __name__ == '__main__':
    app.run(debug=True)
