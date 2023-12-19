# Preprocess data
**keys**
column = ['author_id', 'conversation_id', 'user_id', 'reference_id', 'tweet_id', 'type', 'possibly_sensitive', 'lang', 
           'reply_settings', 'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 
           'impression_count', 'text', 'context', 'mentions', 'annotations', 'urls']
           
1. conversation_id, reference_id, tweet_id: use map_id to map real ids to fake ones
2. possibly_sensitive: 0 if False nad 1 if True
3. type: 0 if 'replied to' and 1 if 'quote'
4. lang: refer to map_lan
5. reply settings: 0 for everyone and 1 for followers