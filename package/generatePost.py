from package.tech import sentimentAnalysis as SA
from package.tech import reinforcementLearningAlgorithm as RLA



def createNewPost(Post, agent):
    
    
    
    newPost = Post(title = "title", content = "content", author = agent)
    return(newPost)




def updatePosts(Post, page, db, agent):
    
    
    currentPosts = Post.query.order_by(Post.date_posted.desc()).paginate(page, per_page = 10)
    newPost = createNewPost(Post)
    
    db.session.add(newPost)
    db.session.commit()
    
    return()