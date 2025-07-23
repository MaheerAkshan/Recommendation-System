import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class RecommendationSystem:
    def __init__(self):
        # Expanded sample data
        self.users = {
            'Alice': {'Action': 5, 'Comedy': 3, 'Drama': 4, 'Horror': 1, 'Sci-Fi': 4},
            'Bob': {'Action': 4, 'Comedy': 2, 'Drama': 5, 'Horror': 3, 'Sci-Fi': 3},
            'Charlie': {'Action': 2, 'Comedy': 5, 'Drama': 1, 'Horror': 4, 'Sci-Fi': 2},
            'Diana': {'Action': 3, 'Comedy': 4, 'Drama': 2, 'Horror': 5, 'Sci-Fi': 3},
            'Eve': {'Action': 5, 'Comedy': 1, 'Drama': 3, 'Horror': 2, 'Sci-Fi': 5}
        }
        
        self.movies = {
            'Movie1': {'title': 'The Dark Knight', 'genres': 'Action,Crime,Drama', 'rating': 9.0, 'year': 2008},
            'Movie2': {'title': 'Pulp Fiction', 'genres': 'Crime,Drama', 'rating': 8.9, 'year': 1994},
            'Movie3': {'title': 'The Shawshank Redemption', 'genres': 'Drama', 'rating': 9.3, 'year': 1994},
            'Movie4': {'title': 'The Godfather', 'genres': 'Crime,Drama', 'rating': 9.2, 'year': 1972},
            'Movie5': {'title': 'Inception', 'genres': 'Action,Adventure,Sci-Fi', 'rating': 8.8, 'year': 2010},
            'Movie6': {'title': 'The Hangover', 'genres': 'Comedy', 'rating': 7.7, 'year': 2009},
            'Movie7': {'title': 'Superbad', 'genres': 'Comedy', 'rating': 7.6, 'year': 2007},
            'Movie8': {'title': 'The Exorcist', 'genres': 'Horror', 'rating': 8.0, 'year': 1973},
            'Movie9': {'title': 'Hereditary', 'genres': 'Horror,Mystery', 'rating': 7.3, 'year': 2018},
            'Movie10': {'title': 'Mad Max: Fury Road', 'genres': 'Action,Adventure,Sci-Fi', 'rating': 8.1, 'year': 2015},
            'Movie11': {'title': 'Interstellar', 'genres': 'Adventure,Drama,Sci-Fi', 'rating': 8.6, 'year': 2014},
            'Movie12': {'title': 'Parasite', 'genres': 'Comedy,Drama,Thriller', 'rating': 8.6, 'year': 2019},
            'Movie13': {'title': 'Joker', 'genres': 'Crime,Drama,Thriller', 'rating': 8.4, 'year': 2019},
            'Movie14': {'title': 'The Matrix', 'genres': 'Action,Sci-Fi', 'rating': 8.7, 'year': 1999},
            'Movie15': {'title': 'Get Out', 'genres': 'Horror,Mystery,Thriller', 'rating': 7.7, 'year': 2017}
        }
        
        self.user_ratings = {
            'Alice': {'Movie1': 5, 'Movie3': 4, 'Movie6': 2, 'Movie11': 5, 'Movie14': 4},
            'Bob': {'Movie2': 5, 'Movie4': 5, 'Movie8': 4, 'Movie13': 3, 'Movie15': 4},
            'Charlie': {'Movie6': 5, 'Movie7': 4, 'Movie10': 2, 'Movie12': 5},
            'Diana': {'Movie8': 5, 'Movie9': 4, 'Movie5': 3, 'Movie15': 5},
            'Eve': {'Movie5': 5, 'Movie11': 5, 'Movie14': 5, 'Movie1': 4}
        }
        
        # Prepare content-based features
        self._prepare_content_features()
        
    def _prepare_content_features(self):
        """Prepare TF-IDF vectors for content-based filtering"""
        movie_descriptions = []
        self.movie_ids = []
        
        for movie_id, movie_data in self.movies.items():
            self.movie_ids.append(movie_id)
            movie_descriptions.append(f"{movie_data['title']} {movie_data['genres']} {movie_data['year']}")
        
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(movie_descriptions)
        self.content_similarity = cosine_similarity(self.tfidf_matrix)
    
    def get_user_preferences(self, username):
        """Get a user's genre preferences"""
        return self.users.get(username, {})
    
    def get_similar_users(self, username, n=3):
        """Find similar users using collaborative filtering"""
        if username not in self.users:
            return []
            
        # Create user vectors
        all_genres = set()
        for user in self.users.values():
            all_genres.update(user.keys())
        all_genres = sorted(all_genres)
        
        # Build user matrix
        user_matrix = []
        user_names = []
        for user, prefs in self.users.items():
            user_names.append(user)
            user_vector = [prefs.get(genre, 0) for genre in all_genres]
            user_matrix.append(user_vector)
        
        # Calculate similarity
        user_sim = cosine_similarity(user_matrix)
        user_index = user_names.index(username)
        
        # Get top N similar users (excluding self)
        similar_users = []
        sim_scores = list(enumerate(user_sim[user_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        
        for i, score in sim_scores:
            similar_users.append((user_names[i], score))
        
        return similar_users
    
    def content_based_recommendations(self, movie_id, n=5):
        """Get content-based recommendations for a movie"""
        if movie_id not in self.movie_ids:
            return []
            
        idx = self.movie_ids.index(movie_id)
        sim_scores = list(enumerate(self.content_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        
        recommendations = []
        for i, score in sim_scores:
            recommendations.append((self.movie_ids[i], score))
        
        return recommendations
    
    def get_movies_by_genre(self, genre, n=5):
        """Get movies by genre"""
        genre_movies = []
        for movie_id, movie_data in self.movies.items():
            if genre.lower() in movie_data['genres'].lower():
                genre_movies.append((movie_id, movie_data['rating']))
        
        # Sort by rating
        genre_movies.sort(key=lambda x: x[1], reverse=True)
        return genre_movies[:n]
    
    def hybrid_recommendations(self, username, n=5):
        """Generate hybrid recommendations combining collaborative and content-based filtering"""
        if username not in self.users:
            return []
            
        # Step 1: Get movies liked by similar users (collaborative)
        similar_users = self.get_similar_users(username)
        collaborative_candidates = set()
        
        for user, score in similar_users:
            for movie_id, rating in self.user_ratings.get(user, {}).items():
                if rating >= 4:
                    collaborative_candidates.add(movie_id)
        
        # Step 2: Get content-based recommendations for user's liked movies
        content_candidates = {}
        user_movies = self.user_ratings.get(username, {})
        
        for movie_id, rating in user_movies.items():
            if rating >= 4:
                similar_movies = self.content_based_recommendations(movie_id)
                for similar_id, score in similar_movies:
                    if similar_id not in user_movies:
                        content_candidates[similar_id] = content_candidates.get(similar_id, 0) + score
        
        all_candidates = {}
        
        # Add collaborative candidates
        for movie_id in collaborative_candidates:
            if movie_id not in user_movies:
                all_candidates[movie_id] = all_candidates.get(movie_id, 0) + 1
        
        # Add content candidates with weight
        for movie_id, score in content_candidates.items():
            all_candidates[movie_id] = all_candidates.get(movie_id, 0) + score * 0.7
        
        # Sort by recommendation score
        sorted_recommendations = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Prepare final recommendations with movie details
        final_recommendations = []
        for movie_id, score in sorted_recommendations:
            movie_data = self.movies.get(movie_id, {})
            final_recommendations.append({
                'movie_id': movie_id,
                'title': movie_data.get('title', 'Unknown'),
                'genres': movie_data.get('genres', ''),
                'year': movie_data.get('year', ''),
                'rating': movie_data.get('rating', 0),
                'recommendation_score': round(score, 2)
            })
        
        return final_recommendations

    def get_top_rated_movies(self, n=5):
        """Get top rated movies overall (cold start solution)"""
        sorted_movies = sorted(self.movies.items(), 
                             key=lambda x: x[1]['rating'], 
                             reverse=True)[:n]
        return [{
            'movie_id': movie_id,
            'title': data['title'],
            'genres': data['genres'],
            'year': data['year'],
            'rating': data['rating']
        } for movie_id, data in sorted_movies]

    def get_user_rated_movies(self, username):
        """Get movies rated by a user"""
        if username not in self.user_ratings:
            return []
        
        rated_movies = []
        for movie_id, rating in self.user_ratings[username].items():
            movie_data = self.movies.get(movie_id, {})
            rated_movies.append({
                'movie_id': movie_id,
                'title': movie_data.get('title', 'Unknown'),
                'genres': movie_data.get('genres', ''),
                'year': movie_data.get('year', ''),
                'rating': movie_data.get('rating', 0),
                'user_rating': rating
            })
        
        return sorted(rated_movies, key=lambda x: x['user_rating'], reverse=True)

def display_landing_page():
    print("\n" + "="*60)
    print(" " * 20 + "MOVIE RECOMMENDATION SYSTEM")
    print("="*60)
    print("\nWelcome to our advanced movie recommendation platform!")
    print("Discover your next favorite movie with our hybrid recommendation engine.")
    print("\nMain Features:")
    print("- Personalized movie recommendations")
    print("- Genre-specific movie browsing")
    print("- Top-rated movies collection")
    print("- User profile insights")
    print("- Similar users analysis")
    print("\nAvailable users:", ", ".join(rs.users.keys()))

def display_movie_details(movie):
    print(f"\nTitle: {movie['title']} ({movie['year']})")
    print(f"Genres: {movie['genres']}")
    print(f"Rating: {movie['rating']}/10")
    if 'user_rating' in movie:
        print(f"Your Rating: {movie['user_rating']}/5")
    if 'recommendation_score' in movie:
        print(f"Recommendation Score: {movie['recommendation_score']:.2f}")

def display_menu(username):
    print("\n" + "="*60)
    print(f"USER DASHBOARD: {username}")
    print("="*60)
    print("\nMain Menu:")
    print("1. Get personalized recommendations")
    print("2. Browse movies by genre")
    print("3. View top-rated movies")
    print("4. View my rated movies")
    print("5. Find similar users")
    print("6. Switch user")
    print("7. Exit")
    return input("\nEnter your choice (1-7): ")

def genre_menu():
    print("\nAvailable genres:")
    genres = set()
    for movie in rs.movies.values():
        for genre in movie['genres'].split(','):
            genres.add(genre.strip())
    for i, genre in enumerate(sorted(genres), 1):
        print(f"{i}. {genre}")
    print(f"{len(genres)+1}. Back to main menu")
    choice = input("\nSelect a genre (or enter 0 to go back): ")
    return choice

rs = RecommendationSystem()

def main():
    current_user = None
    
    while True:
        if current_user is None:
            display_landing_page()
            username = input("\nEnter your username (or 'exit' to quit): ")
            if username.lower() == 'exit':
                break
            if username in rs.users:
                current_user = username
            else:
                print("\nUser not found. Showing top rated movies instead:")
                movies = rs.get_top_rated_movies(5)
                for movie in movies:
                    display_movie_details(movie)
                input("\nPress Enter to continue...")
                continue
        else:
            choice = display_menu(current_user)
            
            if choice == '1':
                print("\nGenerating personalized recommendations...")
                recommendations = rs.hybrid_recommendations(current_user, 5)
                if recommendations:
                    print("\nWe think you'll love these movies:")
                    for i, movie in enumerate(recommendations, 1):
                        print(f"\n{i}. ", end="")
                        display_movie_details(movie)
                else:
                    print("\nNot enough data for personalized recommendations. Showing top rated movies:")
                    movies = rs.get_top_rated_movies(5)
                    for movie in movies:
                        display_movie_details(movie)
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                while True:
                    genre_choice = genre_menu()
                    if genre_choice == '0':
                        break
                    try:
                        genre_num = int(genre_choice)
                        genres = sorted(set(g for movie in rs.movies.values() for g in movie['genres'].split(',')))
                        if 1 <= genre_num <= len(genres):
                            selected_genre = genres[genre_num-1]
                            print(f"\nTop {selected_genre} movies:")
                            movies = rs.get_movies_by_genre(selected_genre, 5)
                            for i, (movie_id, rating) in enumerate(movies, 1):
                                movie = rs.movies[movie_id]
                                print(f"\n{i}. ", end="")
                                display_movie_details(movie)
                            input("\nPress Enter to continue...")
                        else:
                            break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                        continue
                
            elif choice == '3':
                print("\nTop Rated Movies of All Time:")
                movies = rs.get_top_rated_movies(5)
                for i, movie in enumerate(movies, 1):
                    print(f"\n{i}. ", end="")
                    display_movie_details(movie)
                input("\nPress Enter to continue...")
                
            elif choice == '4':
                print(f"\nMovies you've rated, {current_user}:")
                movies = rs.get_user_rated_movies(current_user)
                if movies:
                    for i, movie in enumerate(movies, 1):
                        print(f"\n{i}. ", end="")
                        display_movie_details(movie)
                else:
                    print("You haven't rated any movies yet.")
                input("\nPress Enter to continue...")
                
            elif choice == '5':
                print(f"\nUsers with similar tastes to {current_user}:")
                similar_users = rs.get_similar_users(current_user)
                if similar_users:
                    for i, (user, score) in enumerate(similar_users, 1):
                        print(f"\n{i}. {user} (similarity score: {score:.2f})")
                        print("   Top rated movies:")
                        user_movies = rs.get_user_rated_movies(user)[:3]
                        for movie in user_movies:
                            print(f"   - {movie['title']} ({movie['user_rating']}/5)")
                else:
                    print("No similar users found.")
                input("\nPress Enter to continue...")
                
            elif choice == '6':
                current_user = None
                
            elif choice == '7':
                break
                
            else:
                print("Invalid choice. Please enter a number between 1-7.")
                input("Press Enter to continue...")

    print("\nThank you for using our Movie Recommendation System!")

if __name__ == "__main__":
    main()
