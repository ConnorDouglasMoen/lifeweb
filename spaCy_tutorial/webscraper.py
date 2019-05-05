from selenium import webdriver


names = ['Smith', 'Jones', 'Johnson', 'Moen', 'Chen', 'Alwan']
browser = webdriver.Firefox(executable_path=r'./geckodriver.exe')

# Starting from
for name in names:
    browser.get('https://www.forevermissed.com/findmemorial/?q=' + name)

# # Search names from search box
#     searchEle = browser.find_element_by_name('q')
#     searchEle.send_keys(name)
#     searchEle.submit()
#   ## we might have to worry about it needing to refresh idkk

# for each "fmi-earth" class element, follow the embedded link

# then find all "txt user_text" elements, copy their contents and export them
# to a .txt file
