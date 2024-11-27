import umap
import altair as alt

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def umap_plot(text, emb):

    cols = list(text.columns)
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = umap.UMAP(n_neighbors=2)
    umap_embeds = reducer.fit_transform(emb)
    # Prepare the data to plot and interactive visualization
    # using Altair
    #df_explore = pd.DataFrame(data={'text': qa['text']})
    #print(df_explore)
    
    #df_explore = pd.DataFrame(data={'text': qa_df[0]})
    df_explore = text.copy()
    df_explore['x'] = umap_embeds[:,0]
    df_explore['y'] = umap_embeds[:,1]
    
    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=#'x',
        alt.X('x',
            scale=alt.Scale(zero=False)
        ),
        y=
        alt.Y('y',
            scale=alt.Scale(zero=False)
        ),
        tooltip=cols
        #tooltip=['text']
    ).properties(
        width=700,
        height=400
    )
    return chart

def umap_plot_big(text, emb):

    cols = list(text.columns)
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = umap.UMAP(n_neighbors=100)
    umap_embeds = reducer.fit_transform(emb)
    # Prepare the data to plot and interactive visualization
    # using Altair
    #df_explore = pd.DataFrame(data={'text': qa['text']})
    #print(df_explore)
    
    #df_explore = pd.DataFrame(data={'text': qa_df[0]})
    df_explore = text.copy()
    df_explore['x'] = umap_embeds[:,0]
    df_explore['y'] = umap_embeds[:,1]
    
    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=#'x',
        alt.X('x',
            scale=alt.Scale(zero=False)
        ),
        y=
        alt.Y('y',
            scale=alt.Scale(zero=False)
        ),
        tooltip=cols
        #tooltip=['text']
    ).properties(
        width=700,
        height=400
    )
    return chart

def umap_plot_old(sentences, emb):
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = umap.UMAP(n_neighbors=2)
    umap_embeds = reducer.fit_transform(emb)
    # Prepare the data to plot and interactive visualization
    # using Altair
    #df_explore = pd.DataFrame(data={'text': qa['text']})
    #print(df_explore)
    
    #df_explore = pd.DataFrame(data={'text': qa_df[0]})
    df_explore = sentences
    df_explore['x'] = umap_embeds[:,0]
    df_explore['y'] = umap_embeds[:,1]
    
    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=#'x',
        alt.X('x',
            scale=alt.Scale(zero=False)
        ),
        y=
        alt.Y('y',
            scale=alt.Scale(zero=False)
        ),
        tooltip=['text']
    ).properties(
        width=700,
        height=400
    )
    return chart


def print_result(result):
    """ Print results with colorful formatting """
    for i, item in enumerate(result):
        print(f'item {i}')
        for key in item.keys():
            print(f"{key}:{item.get(key)}")
            print()
        print()


### Revised version
def keyword_search(query,
                   client,
                   results_lang='en',
                   properties=["title", "url", "text"],
                   num_results=3):
    where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
    }

    response = (
        client.query.get("Articles", properties)
        .with_bm25(
            query=query
        )
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
    )

    result = response['data']['Get']['Articles']
    return result


def dense_retrieval(query,
                    client,
                    results_lang='en',
                    properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                    num_results=5):
    nearText = {"concepts": [query]}

    # To filter by language
    where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
    }
    response = (
        client.query
        .get("Articles", properties)
        .with_near_text(nearText)
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
    )

    result = response['data']['Get']['Articles']

    return result


def search_wikipedia_subset(client, query, num_results=3, results_lang='en',
                            properties=["text", "title", "url", "views", "lang", "_additional {distance}"]):
    nearText = {"concepts": [query]}

    # To filter by language
    if results_lang:
        where_filter = {
            "path": ["lang"],
            "operator": "Equal",
            "valueString": results_lang
        }
        response = (
            client.query
            .get("Articles", properties)
            .with_where(where_filter)
            .with_near_text(nearText)
            .with_limit(5)
            .do()
        )

    # Search all languages
    else:
        response = (
            client.query
            .get("Articles", properties)
            .with_near_text(nearText)
            .with_limit(5)
            .do()
        )

    result = response['data']['Get']['Articles']

    return result


def generate_given_context(query, weav_client, co_client):
    results = search_wikipedia_subset(weav_client, query, results_lang='en')

    title = results[0]['title']
    context = results[0]['text']

    prompt = f"""
    You are a useful AI trained to answer questions based on the context your are provided.
    Use the Context Information provided below to answer the questions "{query}". If the answer to 
    the question is in the context, extract it and print it. If it's not contained in the provided 
    information, say "I do not know". 
    ---
    Context information about {title}:
    Context: {context}
    End of Context Information
    ---
    Question: {query}
    """

    # to answer from the Context Information
    prediction = co_client.generate(
        prompt=prompt,
        max_tokens=50,
        # model='command-light',
        # temperature=0.3,
        num_generations=5)

    return prediction, context_title, context_text