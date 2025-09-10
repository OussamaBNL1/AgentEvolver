import copy
from beyondagent.module.task_manager.user_profiles import EnvEntity, EnvEntityOpt, TaskPreference, UserProfile


venmo = EnvEntity(
    name="Venmo",
    description="A mobile payment service to send, request, and manage money transactions.",
    attrs={
        "Account": "User's Venmo account details.",
        "Friends": "List of Venmo friends.",
        "Transactions": "History of sent and received payments."
    },
    opts=[
        EnvEntityOpt("accept_request", "Accept pending payment requests."),
        EnvEntityOpt("reject_request", "Reject payment requests."),
        EnvEntityOpt("send_money", "Send money to a user."),
        EnvEntityOpt("request_money", "Request money from a user."),
        EnvEntityOpt("like_transaction", "Like a Venmo transaction."),
        EnvEntityOpt("remind_request", "Send reminders for pending requests."),
        EnvEntityOpt("manage_friends", "Befriend or unfriend users on Venmo."),
        EnvEntityOpt("change_password", "Change account password.")
    ]
)

amazon = EnvEntity(
    name="Amazon",
    description="An online e-commerce platform for purchasing, returning, and reviewing products.",
    attrs={
        "Account": "User's Amazon account details.",
        "Cart": "Items currently in the shopping cart.",
        "Wishlist": "Products saved for later purchase.",
        "Orders": "Purchase history and delivery details."
    },
    opts=[
        EnvEntityOpt("place_order", "Place an order for products."),
        EnvEntityOpt("return_item", "Initiate a return process."),
        EnvEntityOpt("manage_cart", "Add, remove, or move items in the cart."),
        EnvEntityOpt("manage_wishlist", "Add or remove items from wishlist."),
        EnvEntityOpt("post_question", "Post a product-related question."),
        EnvEntityOpt("write_review", "Write or update a product review."),
        EnvEntityOpt("check_delivery", "Check delivery date for an order.")
    ]
)

spotify = EnvEntity(
    name="Spotify",
    description="A music streaming service with song, album, and playlist management.",
    attrs={
        "Song Library": "Songs saved by the user.",
        "Album Library": "Albums saved by the user.",
        "Playlists": "User-created or followed playlists."
    },
    opts=[
        EnvEntityOpt("play_song", "Play a song, album, or playlist."),
        EnvEntityOpt("like_song", "Like songs or albums."),
        EnvEntityOpt("unfollow_artist", "Unfollow an artist."),
        EnvEntityOpt("follow_artist", "Follow an artist."),
        EnvEntityOpt("create_playlist", "Create a new playlist."),
        EnvEntityOpt("remove_song", "Remove songs from library or playlist."),
        EnvEntityOpt("export_library", "Export song/album/playlist data.")
    ]
)

gmail = EnvEntity(
    name="Gmail",
    description="An email service for sending, receiving, labeling, and managing emails.",
    attrs={
        "Inbox": "List of received email threads.",
        "Outbox": "List of sent email threads.",
        "Labels": "Custom labels to organize emails."
    },
    opts=[
        EnvEntityOpt("send_email", "Send an email."),
        EnvEntityOpt("forward_email", "Forward an email."),
        EnvEntityOpt("reply_email", "Reply to an email."),
        EnvEntityOpt("delete_email", "Delete emails."),
        EnvEntityOpt("label_email", "Label emails."),
        EnvEntityOpt("star_email", "Star or unstar email threads.")
    ]
)

simplenote = EnvEntity(
    name="Simple Note",
    description="A note-taking app for storing and managing notes and lists.",
    attrs={
        "Notes": "Collection of user notes.",
        "Tags": "Tags for organizing notes."
    },
    opts=[
        EnvEntityOpt("export_note", "Export notes."),
        EnvEntityOpt("update_note", "Update or edit a note."),
        EnvEntityOpt("add_note", "Add a new note."),
        EnvEntityOpt("note_to_playlist", "Create a playlist from note content.")
    ]
)

phone = EnvEntity(
    name="Phone",
    description="A mobile device for calls, text messages, voice messages, and alarms.",
    attrs={
        "Contacts": "List of saved contacts.",
        "Messages": "Text and voice messages.",
        "Alarms": "Configured alarms on the device."
    },
    opts=[
        EnvEntityOpt("send_text", "Send a text message."),
        EnvEntityOpt("send_voice", "Send a voice message."),
        EnvEntityOpt("set_alarm", "Set or update an alarm.")
    ]
)

todoist = EnvEntity(
    name="Todoist",
    description="A task management and to-do list application.",
    attrs={
        "Projects": "User's task projects.",
        "Tasks": "Individual tasks in projects."
    },
    opts=[
        EnvEntityOpt("complete_task", "Complete tasks."),
        EnvEntityOpt("update_task", "Update tasks."),
        EnvEntityOpt("move_task", "Move tasks between projects.")
    ]
)

splitwise = EnvEntity(
    name="Splitwise",
    description="An app for tracking shared expenses and balances.",
    attrs={
        "Groups": "Expense sharing groups.",
        "Expenses": "List of shared expenses."
    },
    opts=[
        EnvEntityOpt("add_expense", "Add a shared expense."),
        EnvEntityOpt("settle_expense", "Settle or record payments.")
    ]
)

filesystem = EnvEntity(
    name="File System",
    description="A local file storage system for managing files and directories.",
    attrs={
        "Directories": "Folder structure for files.",
        "Files": "Stored documents, images, and other file types."
    },
    opts=[
        EnvEntityOpt("download_file", "Download a file."),
        EnvEntityOpt("move_file", "Move files."),
        EnvEntityOpt("compress_file", "Compress files."),
        EnvEntityOpt("delete_file", "Delete files."),
        EnvEntityOpt("reorganize_files", "Change file organization structure.")
    ]
)




user_profile = UserProfile(
    name="Bob",
    background="A general computer user.",
    task=TaskPreference(
        num_entities=2,
        num_opts=3,
        relation_difficulty=3,
    )
)

user_profile.reg_rubric("""You must follow these pattern to generate query:
# Venmo
Accept / Approve / Reject all pending Venmo requests [from <groups>] [in <timeframe>].
Send / Request / Remind money on Venmo with note "<message>" [public/private] [amount <X>] [to/from <person/group>].
Comment "<message>" and/or Like Venmo transactions [from <groups>] [in <timeframe>].
Add / Remove Venmo friends to match <groups> [if they are not already / reset from phone contacts].
Calculate totals: [how much money sent/received/requested, how many likes, how many new friends] [in <timeframe>].
Correct / Delete and Recreate mistaken Venmo requests with adjusted amount.

# Amazon
Buy <quantity> <product> on Amazon with constraints [rating, reviews, price, seller, dimensions, delivery time, wishlist/cart].
Buy identical gifts for <relations>, with gift wrapping, delivered individually.
Return Amazon orders/items [prefer <carrier>] and forward confirmation to requester.
Change Amazon review [rating, title].
Post question on last Amazon order "<question>".
Download/organize Amazon receipts into "<folder>/<pattern>". Compute total spend [in <timeframe>].
Compare promo codes (cart vs email) and place cheapest order.
Move products between cart and wishlist based on <filters>.
Place grouped Amazon orders from checklist/shopping list in SimpleNote or email.

# Spotify
Play song(s) from Spotify [most/least played, released in <year>, by <artist>, until <condition>].
Like / Unfollow / Rate (stars) songs or albums, adjust ratings if <condition>.
Follow artists: [by genre, recommendation lists, liked songs].
Add / Remove songs in playlists based on external lists (notes, files, email).
Create new playlist "<title>" from [liked songs, SimpleNote entries, top per playlist/album].
Export/Backup Spotify libraries to CSV [with Title, Artists], optionally terminate account.
Count/Report: [playlist duration, unique songs, songs by year, top genres, top/least artists].
Cleanup libraries: keep only liked/downloaded songs/albums, remove others.
Queue management: reset/shuffle queue, add recommendations, remove liked/unliked.

# SimpleNote / Todoist
Habit tracking: add today’s entry based on yesterday [with modifications].  
Export logs to CSV [date + habits yes/no, ordered].  
Compute longest streak of <habit>.
Bucket list: mark items done/not done, count remaining/completed.
Job search: use cover letter + resume from file system, send applications. Schedule reminders.  
Recruiting: parse applicant emails into CSV, download attachments, reply with templates.
Tasks: reassign based on comments, leave acknowledgment. Nightly flow: move tasks Inbox → Today project. Complete trip tasks.

# Gmail
Auto-reply / Reminder emails: set body "<message>", subject "<pattern>", <timeframe>.  
Forward email/thread with prefixed note and/or attachments.  
Attach correct files (cv/resume/headshot, bills, tickets) and send.  
Archive/Delete/Label/Star/Unstar emails by <conditions>.  
Parse bills/expenses/promos from Gmail and save files/CSV.  
Group/birthday/company reminder emails auto-scheduled.

# Filesystem
Delete files matching <extension>.  
Reorganize directories: move/group photos by vacation, rename files with <date> prefix.  
Compress subdirectories into archive (<zip/tar>), delete originals.  
Reorganize meeting files from "<date>__<file>" → "<file>/<date>".  
Add photos to received archives and resend with new subject/body.

# Phone (SMS/Voice)
Send text/voice "<message>" to <relations>.  
Reject spam messages/calls from <numbers>.  
Check phone messages for details (shopping orders, recommendations) and perform.  
Update/shared credentials via phone text.

# Alarms
Create/Update phone alarms:  
- For wake/sleep with <snooze/time change>.  
- For flights/events <X> minutes before.  
- For weekly standup/meeting schedules sent via email.

# Queries / Stats
“How much / how many / have I … ?” across Venmo, Amazon, Spotify, Gmail, bills, subscriptions.  
“Which/What is … ?” → most/least/newest/oldest songs, genres, artists, photos, costs.                        
""")


user_profile.reg_entities([venmo, amazon, spotify, gmail, simplenote, phone, todoist, splitwise, filesystem])

user_profile_wo_rubric=copy.deepcopy(user_profile)
user_profile_wo_rubric._rubrics.clear() # clear the rubrics